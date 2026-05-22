train:
    train \
        --train-data-path "data/CC15" \
        --clean-keyword "_mean" \
        --noisy-keyword "_real" \
        --batch-size 4 \
        --iterations 50 \
        --interval 10 \
        --output-dir "./results" \
