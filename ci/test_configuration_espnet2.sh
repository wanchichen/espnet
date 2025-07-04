#!/usr/bin/env bash

set -euo pipefail

source tools/activate_python.sh
PYTHONPATH="${PYTHONPATH:-}:$(pwd)/tools/s3prl"
export PYTHONPATH
python="coverage run --append"

gen_dummy_coverage(){
    # To avoid a problem when parallel running for `coverage run`.
    # Please put this command after cd ./egs2/foo/bar
    touch empty.py; ${python} empty.py
}

#### Make sure chainer-independent ####
python3 -m pip uninstall -y chainer

# [ESPnet2] Validate configuration files
echo "<blank>" > dummy_token_list
echo "==== [ESPnet2] Validation configuration files ==="
if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.8.0")' &> /dev/null;  then

    s3prl_confs='[ "egs2/fsc/asr1/conf/train_asr.yaml",
        "egs2/americasnlp22/asr1/conf/train_asr_transformer.yaml",
        "egs2/aphasiabank/asr1/conf/train_asr.yaml".
        "egs2/bur_openslr80/asr1/conf/train_asr_hubert_transformer_adam_specaug.yaml",
        "egs2/catslu/asr1/conf/train_asr.yaml",
        "egs2/dcase22_task1/asr1/conf/train_asr.yaml",
        "egs2/fleurs/asr1/conf/train_asr.yaml",
        "egs2/fsc_challenge/asr1/conf/train_asr.yaml",
        "egs2/fsc_unseen/asr1/conf/train_asr.yaml",
        "egs2/meld/asr1/conf/train_asr.yaml",
        "egs2/microsoft_speech/asr1/conf/train_asr.yaml",
        "egs2/mini_an4/asr1/conf/train_asr_transducer_debug.yaml",
        "egs2/slue-voxceleb/asr1/conf/train_asr.yaml",
        "egs2/slue-voxpopuli/asr1/conf/train_asr.yaml",
        "egs2/stop/asr1/conf/train_asr2_hubert_lr0.002.yaml",
        "egs2/stop/asr1/conf/train_asr2_wav2vec2_lr0.002.yaml",
        "egs2/stop/asr1/conf/train_asr2_wavlm_branchformer.yaml",
        "egs2/stop/asr1/conf/train_asr2_wavlm_lr0.002.yaml",
        "egs2/swbd_da/asr1/conf/train_asr.yaml",
        "egs2/totonac/asr1/conf/train_asr.yaml" ]'

    warprnnt_confs='[
        "egs2/librispeech/asr1/conf/train_asr_rnnt.yaml",
     ]'

    for f in egs2/*/asr1/conf/train_asr*.yaml; do
        if [[ ${s3prl_confs} =~ \"${f}\" ]]; then
            if ! python3 -c "import s3prl" &> /dev/null; then
                continue
            fi
        fi
        if [[ ${warprnnt_confs} =~ \"${f}\" ]]; then
            if ! python3 -c "from warprnnt_pytorch import RNNTLoss" &> /dev/null; then
                continue
            fi
        fi
        if [ "$f" == "egs2/how2_2000h/asr1/conf/train_asr_conformer_lf.yaml" ]; then
            if ! python3 -c "import longformer" > /dev/null; then
                continue
            fi
        fi
        if [ "$f" == "egs2/stop/asr1/conf/train_asr_whisper_full_correct.yaml" ]; then
            if ! python3 -c "import whisper" > /dev/null; then
                continue
            fi
        fi
        if [ "$f" == "egs2/uslu14/asr1/conf/train_asr_whisper_full_correct_specaug.yaml" ]; then
            if ! python3 -c "import whisper" > /dev/null; then
                continue
            fi
        fi
        ${python} -m espnet2.bin.asr_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
        sudo rm -rf /root/.cache/huggingface*
        rm -rf hf_cache hub
    done

    for f in egs2/*/asr1/conf/train_transducer*.yaml; do
        ${python} -m espnet2.bin.asr_transducer_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
    done

    for f in egs2/*/asr1/conf/train_lm*.yaml; do
        ${python} -m espnet2.bin.lm_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
    done

    for f in egs2/*/tts1/conf/train*.yaml; do
        ${python} -m espnet2.bin.tts_train --config "${f}" --iterator_type none --normalize none --dry_run true --output_dir out --token_list dummy_token_list
    done

    for f in egs2/*/enh1/conf/train*.yaml; do
        ${python} -m espnet2.bin.enh_train --config "${f}" --iterator_type none --dry_run true --output_dir out
    done

    if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.12.0")' &> /dev/null; then
        for f in egs2/*/ssl1/conf/train*.yaml; do
            ${python} -m espnet2.bin.hubert_train --config "${f}" --iterator_type none --normalize none --dry_run true --output_dir out --token_list dummy_token_list --num_classes 10
        done
    fi

    for f in egs2/*/enh_asr1/conf/train_enh_asr*.yaml; do
        ${python} -m espnet2.bin.enh_s2t_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
    done
fi

# These files must be same each other.
for base in cmd.sh conf/slurm.conf conf/queue.conf conf/pbs.conf; do
    file1=
    for f in egs2/*/*/"${base}"; do
        if [ -z "${file1}" ]; then
            file1="${f}"
        fi
        diff "${file1}" "${f}" || { echo "Error: ${file1} and ${f} differ: To solve: for f in egs2/*/*/${base}; do cp egs2/TEMPLATE/asr1/${base} \${f}; done" ; exit 1; }
    done
done


echo "==== [ESPnet2] test setup.sh ==="
for d in egs2/TEMPLATE/*; do
    if [ -d "${d}" ]; then
        d="${d##*/}"
        egs2/TEMPLATE/"$d"/setup.sh egs2/test/"${d}"
    fi
done
