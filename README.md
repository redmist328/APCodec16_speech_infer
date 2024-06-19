## Description
The model is trained on public speech datasets (maybe not works well on audio).

## Requirements
Follow this [txt](https://github.com/redmist328/APNet2/blob/main/requirements.txt).

## Inference
You can download pretrained model at [here](https://github.com/redmist328/APCodec16_speech_infer/tree/main/requirements). And put the files in [cp_Encoder_Decoder](https://github.com/redmist328/APCodec16_speech_infer/tree/main/cp_Encoder_Decoder).

There are 2 kinds of bitrate can be seleceted: 2 kbps and 4kbps.

You need to change the "n_codebooks" in config.json to change the bitrate, 4 for 2kbps and 8 for 4 kbps. Then change the "checkpoint_file_load_xx" with the same bitrate file.
```
python inference.py
```
