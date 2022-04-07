DATA="NAL_10fold"

for SPLIT in {0..9}
do
  ENTITY=$SPLIT

  echo "***** " $DATA " test score " $SPLIT " *****"
  python re_eval.py \
    --output_path=../dataset/10fold/$ENTITY/test_results.txt \
    --answer_path=../dataset/10fold/$ENTITY/test_original.tsv
done

