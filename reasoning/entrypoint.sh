#!/usr/bin/env bash

# Download extra resources
pip show en_core_web_lg || python -m spacy download en_core_web_lg
python -c "import nltk;nltk.download('punkt')"


maybe_download_model() {
	FID=$REASONING_MODELS_PATH/.${1}.google_drive_id
        [ "$([ -f "$FID" ] && cat "$FID")" != "${2}" ] \
                && download_model $@ && echo -n ${2} > $FID
}

download_model() {
	ZF="$REASONING_MODELS_PATH/${1}.zip"
        gdown "${2}" -O "$ZF" && unzip "$ZF" -d $REASONING_MODELS_PATH && rm "$ZF"
}

maybe_download_model recipe_tagger 1aYSlngadawRTKuIkd1FtMvrMBfenqPLH

python main.py run
