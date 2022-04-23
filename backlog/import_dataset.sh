if [ -f .env ]; then
    # Load Environment Variables
    export $(cat .env | grep -v '#' | sed 's/\r$//' | awk '/=/ {print $1}' )
fi

rm -rf /src/datasets/ifgaudio
mkdir /src/datasets/ifgaudio

scp -r $BCC_USERNAME@$BCC_HOST:/home/anapolis/ifaudio/ifaudio_lumen/database/ifaudio/pessoas/data /src/datasets/ifgaudio/data
scp -r $BCC_USERNAME@$BCC_HOST:/home/anapolis/ifaudio/ifaudio_lumen/uploads /src/datasets/ifgaudio/audio