<!-- Generated by scripts/utils/show_asr_result.sh -->
# RESULTS

# asr_train_asr_conformer_raw_zh_char_sp

## Environments
- date: `Thu Jun 16 16:51:22 CST 2022`
- python version: `3.8.13 (default, Mar 28 2022, 11:38:47)  [GCC 7.5.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.7.0`
- Git hash: `991eaa4a9e22c114ca59ef3988b4fcd0cdf25cdf`
  - Commit date: `Sat Jun 11 14:09:32 2022 +0800`

## Model info
- Model link: https://huggingface.co/espnet/aishell2_att_ctc_espnet2
- ASR config: conf/train_asr_conformer.yaml
- Decode config: conf/decode_asr_rnn.yaml 
- LM config: conf/train_lm_transformer.yaml 

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_asr_model_valid.acc.ave/dev_ios|2500|24802|94.8|5.0|0.2|0.1|5.4|33.7|
|decode_asr_rnn_asr_model_valid.acc.ave/test_android|5000|49534|94.0|5.8|0.2|0.1|6.1|36.2|
|decode_asr_rnn_asr_model_valid.acc.ave/test_ios|5000|49534|94.5|5.4|0.2|0.1|5.7|34.5|
|decode_asr_rnn_asr_model_valid.acc.ave/test_mic|5000|49534|94.0|5.8|0.2|0.1|6.1|36.6|
|decode_asr_rnn_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev_ios|2500|24802|94.9|4.9|0.3|0.1|5.2|31.6|
|decode_asr_rnn_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test_android|5000|49534|94.1|5.6|0.3|0.1|6.0|35.0|
|decode_asr_rnn_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test_ios|5000|49534|94.6|5.1|0.2|0.1|5.5|33.4|
|decode_asr_rnn_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test_mic|5000|49534|94.3|5.5|0.2|0.1|5.8|34.6|


# asr_train_conformer-rnn_transducer_raw_zh_char_sp

## Environments
- date: `Tue Jul  5 22:02:55 CST 2022`
- python version: `3.8.13 (default, Mar 28 2022, 11:38:47)  [GCC 7.5.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.7.1`
- Git hash: `40c5f6919244c2ec8eac14b9011854dd02511a04`
  - Commit date: `Fri Jun 17 11:07:26 2022 +0800`

## Model info
- Model link: https://huggingface.co/espnet/aishell2_transducer
- ASR config: conf/tuning/transducer/train_conformer-rnn_transducer.yaml
- Decode config: conf/tuning/transducer/decode.yaml
- LM config: conf/train_lm_transformer.yaml

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.cer_transducer.ave/dev_ios|2500|24802|94.8|5.0|0.2|0.1|5.3|32.4|
|decode_asr_model_valid.cer_transducer.ave/test_android|5000|49534|94.0|5.7|0.2|0.1|6.1|36.8|
|decode_asr_model_valid.cer_transducer.ave/test_ios|5000|49534|94.8|5.0|0.2|0.1|5.4|33.8|
|decode_asr_model_valid.cer_transducer.ave/test_mic|5000|49534|94.1|5.7|0.2|0.1|6.0|36.1|
|decode_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.cer_transducer.ave/dev_ios|2500|24802|95.1|4.7|0.2|0.1|5.1|31.1|
|decode_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.cer_transducer.ave/test_android|5000|49534|94.2|5.5|0.3|0.1|5.9|35.6|
|decode_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.cer_transducer.ave/test_ios|5000|49534|94.9|4.9|0.2|0.1|5.2|32.6|
|decode_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.cer_transducer.ave/test_mic|5000|49534|94.3|5.4|0.2|0.1|5.8|34.7|