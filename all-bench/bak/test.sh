echo $1
echo $2

if [ -f "run-all.sh" ]; then
  echo "The file exists and is a regular file."
else
  echo "The file does not exist or is not a regular file."
fi
