cd ~
echo "Cloning the code ..."
git clone https://github.com/lamhoangtung/CRAFT-Remade
cd CRAFT-Remade
echo "Downloading SynthText Japanese dataset for strong supervision ..."
cd input
gdrive download 1NcMCuzh1MhgOOQJybxCJcfkNbD_d0w4z
echo "Downloading Datapile dataset for weak supervision ..."
gdrive download 1Ua2KrUkTsxX0Nb5EG17eNgEopbz8fHWL
unzip datapile.zip
echo "Downloading pre-trained weight ..."
cd ../model/
gdrive download 1be4MtJMEcaolM-s4EMsCRUmJFg2pR2OI
gdrive download 1ZQE0tK9498RhLcXwYRgod4upmrYWdgl9
cd ..
echo "Installing dependencies ..."
pip3 install -r requirements.txt
echo "Preprocessing dataset ..."
python3 main.py pre-process --dataset datapile
echo "Everthing is almost done. Check all of yours config and start training weak supervision with this command:"
echo "$ python3 main.py weak-supervision --model ./model/craft_mlt_25k.pth --iterations 10"
echo "Or read the README for more"
