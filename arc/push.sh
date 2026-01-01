zip -r files.zip . \
  -i "*.toml" "*.py" "*.sh" "*.txt" ".env" \
  -x "arc_data/*"
zip -r files.zip arc_data/eval_data
scp -P "$2" files.zip root@"$1":~/files.zip
rm files.zip
