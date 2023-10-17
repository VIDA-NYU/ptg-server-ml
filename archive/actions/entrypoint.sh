#!/usr/bin/env bash

maybe_download_model() {
	FID=$REASONING_MODELS_PATH/.${1}.google_drive_id
        [ "$([ -f "$FID" ] && cat "$FID")" != "${2}" ] \
                && download_model $@ && echo -n ${2} > $FID
}

download_model() {
	ZF="$REASONING_MODELS_PATH/${1}.zip"
        gdown "${2}" -O "$ZF" && unzip "$ZF" -d $REASONING_MODELS_PATH && rm "$ZF"
}

maybe_download_model epic_mir_plus 1Vd8Q-Myj2lgkZrmyImCjZlcJjGpZ3Zff
maybe_download_model egovlp_fewshot 1eqASv2bU1VnVf2qK8_prCn9HNXdK35d3

python main.py run $@
