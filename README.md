## Description
The model is trained on public speech datasets at 16 kHz (maybe not works well on audio).

## Requirements
Follow this [txt](https://github.com/redmist328/APCodec16_speech_infer/tree/main/requirements.txt).

## Inference
You can download pretrained model at [here](http://home.ustc.edu.cn/~redmist/codec/). And put the files in [cp_Encoder_Decoder](https://github.com/redmist328/APCodec16_speech_infer/tree/main/cp_Encoder_Decoder).

There are 2 kinds of bitrate which can be seleceted: 2 kbps and 4kbps.

You need to change the "n_codebooks" in config.json to change the bitrate, n_codebooks=4 for 2kbps and n_codebooks=8 for 4 kbps. Then change the parameter "checkpoint_file_load_xx" with the same bitrate file.
Change "test_input_wavs_dir" in config.json as the input wav dir.
```
python inference.py
```
