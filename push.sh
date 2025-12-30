zip -r files.zip . \
  -i "*.toml" "*.py" "*.sh" "*.txt" "src/data/arc_data_eval/*" \
  -x "*/.*" -x ".*"
scp -P "$2" files.zip root@"$1":~/files.zip
rm files.zip
