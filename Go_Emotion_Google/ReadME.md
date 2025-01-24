annotations_creators:
  - crowdsourced
language_creators:
  - found
language:
  - en
license:
  - apache-2.0
multilinguality:
  - monolingual
size_categories:
  - 100K<n<1M
  - 10K<n<100K
source_datasets:
  - original
task_categories:
  - text-classification
task_ids:
  - multi-class-classification
  - multi-label-classification
paperswithcode_id: goemotions
pretty_name: GoEmotions
config_names:
  - raw
  - simplified
tags:
  - emotion
dataset_info:
  - config_name: raw
    features:
      - name: text
        dtype: string
      - name: id
        dtype: string
      - name: author
        dtype: string
      - name: subreddit
        dtype: string
      - name: link_id
        dtype: string
      - name: parent_id
        dtype: string
      - name: created_utc
        dtype: float32
      - name: rater_id
        dtype: int32
      - name: example_very_unclear
        dtype: bool
      - name: admiration
        dtype: int32
      - name: amusement
        dtype: int32
      - name: anger
        dtype: int32
      - name: annoyance
        dtype: int32
      - name: approval
        dtype: int32
      - name: caring
        dtype: int32
      - name: confusion
        dtype: int32
      - name: curiosity
        dtype: int32
      - name: desire
        dtype: int32
      - name: disappointment
        dtype: int32
      - name: disapproval
        dtype: int32
      - name: disgust
        dtype: int32
      - name: embarrassment
        dtype: int32
      - name: excitement
        dtype: int32
      - name: fear
        dtype: int32
      - name: gratitude
        dtype: int32
      - name: grief
        dtype: int32
      - name: joy
        dtype: int32
      - name: love
        dtype: int32
      - name: nervousness
        dtype: int32
      - name: optimism
        dtype: int32
      - name: pride
        dtype: int32
      - name: realization
        dtype: int32
      - name: relief
        dtype: int32
      - name: remorse
        dtype: int32
      - name: sadness
        dtype: int32
      - name: surprise
        dtype: int32
      - name: neutral
        dtype: int32
    splits:
      - name: train
        num_bytes: 55343102
        num_examples: 211225
    download_size: 24828322
    dataset_size: 55343102
  - config_name: simplified
    features:
      - name: text
        dtype: string
      - name: labels
        sequence:
          class_label:
            names:
              '0': admiration
              '1': amusement
              '2': anger
              '3': annoyance
              '4': approval
              '5': caring
              '6': confusion
              '7': curiosity
              '8': desire
              '9': disappointment
              '10': disapproval
              '11': disgust
              '12': embarrassment
              '13': excitement
              '14': fear
              '15': gratitude
              '16': grief
              '17': joy
              '18': love
              '19': nervousness
              '20': optimism
              '21': pride
              '22': realization
              '23': relief
              '24': remorse
              '25': sadness
              '26': surprise
              '27': neutral
      - name: id
        dtype: string
    splits:
      - name: train
        num_bytes: 4224138
        num_examples: 43410
      - name: validation
        num_bytes: 527119
        num_examples: 5426
      - name: test
        num_bytes: 524443
        num_examples: 5427
    download_size: 3464371
    dataset_size: 5275700
configs:
  - config_name: raw
    data_files:
      - split: train
        path: raw/train-*
  - config_name: simplified
    data_files:
      - split: train
        path: simplified/train-*
      - split: validation
        path: simplified/validation-*
      - split: test
        path: simplified/test-*
    default: true