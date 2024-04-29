

download:
	!gdown "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VJOalBs4I_1f-d5M8g8JRxrNDywgUsqK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VJOalBs4I_1f-d5M8g8JRxrNDywgUsqK" -O journey-springfield.zip && rm -rf /tmp/cookies.txt

zip:
	!unzip journey-springfield.zip -d journey-springfield