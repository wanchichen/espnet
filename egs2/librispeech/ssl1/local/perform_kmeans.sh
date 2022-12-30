#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
train_set=
dev_set=
datadir=dump/raw
feat_dir=dump/hubert_feats
km_dir=
dictdir=
alignment_phoneme_dir=
wave_file_path_prefix=   # The root prefix regarding to items in dump/org/${dset}/wav.scp,  e.g. ${LIBRISPEECH}/LibriSpeech
phn_sets="dev-other dev-clean"
use_gpu=false

nclusters=100
feature_type=mfcc
layer=

# Extract intermediate Hubert embedding from official hubert model:
hubert_type="espnet"  # fairseq or espnet
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models/hubert_base_ls960.pt"

# Extract intermediate Hubert embedding from espnet-trained model:
# hubert_url="espnet"
# hubert_dir_path="" # Pretrained Hubert model dir contains 'valid.acc.best.pth' and 'config.yaml'

portion=0.1
nj=16
python=python3       # Specify python to execute espnet commands.

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 <--nclusters:100> <--feature_type:mfcc>"
    exit 0
fi


km_tag=$(basename ${km_dir})

if [ "${feature_type}" = "hubert" ]; then
    suffix="layer${layer}/"
else
    suffix=""
    use_gpu=false  # mfcc feature does not require GPU.
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preprocess data to generate tsv files."

    for dset in "${train_set}" "${dev_set}"; do
        echo "${wave_file_path_prefix}" > "${datadir}"/${dset}/wav.tsv

        paste "${datadir}/${dset}"/wav.scp "${datadir}/${dset}"/utt2num_samples | \
            awk '{print($2 "\t" $4)}' | sed "s=${wave_file_path_prefix}/==" >> "${datadir}/${dset}"/wav.tsv
    done

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Dump ${feature_type} feature"

    if ${use_gpu}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${train_cmd}"
        _ngpu=0
    fi

    for dset in "${train_set}" "${dev_set}"; do
        echo "${dset}"

        nutt=$(<"${datadir}/${dset}"/wav.scp wc -l)
        _nj=$((nj<nutt?nj:nutt))

        # shellcheck disableSC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${feat_dir}/${feature_type}/${suffix}${dset}"/log/dump_feats.JOB.log \
            ${python} local/dump_mfcc_or_hubert_features.py \
                --data_dir "${datadir}/${dset}" \
                --split "wav" \
                --feat_dir "${feat_dir}/${feature_type}/${suffix}${dset}/data" \
                --feature_type "${feature_type}" \
                --hubert_type "${hubert_type}" \
                --hubert-model-url "${hubert_url}" \
                --hubert-model-path "${hubert_dir_path}" \
                --layer "${layer}" \
                --nshard "${_nj}" \
                --rank "JOB-1" || exit 1;
    done

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Learn K-means with ${feature_type} feature based on scikit-learn"

    mkdir -p "${km_dir}/model"

    nutt=$(<"${datadir}/${train_set}"/wav.scp wc -l)
    _nj=$((nj<nutt?nj:nutt))

    ${train_cmd} ${km_dir}/log/learn_kmeans.log \
        python local/learn_kmeans.py \
            --feat_dir "${feat_dir}/${feature_type}/${suffix}${train_set}/data" \
            --split "wav" \
            --nshard ${_nj} \
            --km_path ${km_dir}/km_${nclusters}.mdl \
            --n_clusters ${nclusters} \
            --percent ${portion} || exit 1;

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Generate K-means pseudo-labels"

    for dset in ${train_set} ${dev_set}; do
        label_dir="${km_dir}/pseudo_labels/${dset}"

        if [[ -d "${label_dir}" ]]; then
            echo "${label_dir} already exists, will remove it"
            rm -r ${label_dir}
        fi
        mkdir -p ${label_dir}

        nutt=$(<"${datadir}/${dset}"/wav.scp wc -l)
        _nj=$((nj<nutt?nj:nutt))

        ${train_cmd} JOB=1:${_nj} ${label_dir}/log/dump_km_label.JOB.log \
            ${python} local/dump_km_label.py \
                --feat_dir "${feat_dir}/${feature_type}/${suffix}${dset}/data" \
                --split "wav" \
                --km_path "${km_dir}/km_${nclusters}.mdl" \
                --nshard ${_nj} \
                --rank "JOB-1" \
                --lab_dir "${label_dir}" || exit 1;

        for rank in $(seq 0 1 $((_nj - 1))); do
            cat ${label_dir}/wav_${rank}_${_nj}.km
        done > ${label_dir}/wav.km

        sed '1d' "${datadir}/${dset}"/wav.tsv | \
            awk '{n=split($1, lst, "/"); uttname=lst[n]; gsub(/\.wav|\.flac/, "", uttname); print(uttname)}' | \
            paste - ${label_dir}/wav.km > ${label_dir}/pseudo_labels.txt

        cp ${label_dir}/pseudo_labels.txt "${datadir}/${dset}"/text.km.${km_tag}

        utils/fix_data_dir.sh --utt_extra_files "text.km.${km_tag}" ${datadir}/${dset}
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Generate char-based fairseq style dictionary: <token> <count>"
    # generate dictionaries
    oov="<unk>"         # Out of vocabulary symbol.
    blank="<blank>"     # CTC blank symbol
    pad="<pad>"
    sos_eos="<sos/eos>" # sos and eos symbole

    label_dir="${km_dir}/pseudo_labels/${train_set}"
    mkdir -p ${dictdir}

    <${label_dir}/pseudo_labels.txt cut -d" " -f2- | \
        awk '{for (i=1; i<=NF; i++) {count[$i]+=1}} END{for (k in count) {print(k, count[k])}}' | \
            sort -n -r -k 2  | \
            awk -v oov=${oov} -v blank=${blank} -v sos_eos=${sos_eos} -v pad=${pad} \
                '{print($1)} END{print(oov); print(sos_eos)}' \
            > ${dictdir}/tokens.txt

    log "Successfully generate the ${dictdir}/{dict,tokens}.txt"

fi

if [ -n "${alignment_phoneme_dir}" ]; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Measure qualities of pseudo labels"

        if [ "${feature_type}" = "hubert" ]; then
            upsample=2
        else
            upsample=1
        fi

        python local/measure_teacher_quality.py \
            --tsv_dir ${datadir} \
            --lab_dir ${datadir} \
            --lab_name "text.km.${km_tag}" \
            --lab_sets "${dev_set}" \
            --phn_dir "${alignment_phoneme_dir}" \
            --phn_sets ${phn_sets} \
            --pad_len 0 \
            --upsample ${upsample} \
            --ref_lab_dir "" \
            --ref_lab_name "" \
            --remove_uid_from_lab true | tee ${km_dir}/pseudo_labels/phoneme_pseudo_label_quality.txt

    fi
fi
