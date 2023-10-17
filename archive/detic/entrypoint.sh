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

maybe_download_model egohos 1m7C_rWymUi045CpWkaXza8zWJ-Emys1X

python main.py run $@
