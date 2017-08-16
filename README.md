# Neural Style Transfer on Audio

This project is the code release for the thesis found in http://barisdemirdelen.com/audio-style-transfer

The code only works on: __Python 3.6__.

------------------


## Dependencies

* python>=3.6
* tensorflow
* librosa
* scipy


------------------


## Usage

In src folder:

```bash
python windowed_style_transfer.py -c <content_wav_file> -s <style_wav_file>
```

The result will be produced in the output/result.wav

------------------
