# Configuration for Sentiment Classification Script

## Model Configuration
You'll need to replace the placeholder model name with your actual quantized Llama model. Here are some popular options:

### Quantized Llama Models (AWQ format):
- `TheBloke/Llama-2-7B-Chat-AWQ`
- `TheBloke/Llama-2-13B-Chat-AWQ`
- `TheBloke/CodeLlama-7B-Instruct-AWQ`
- `casperhansen/llama-3-8b-instruct-awq`

### Quantized Llama Models (GPTQ format):
- `TheBloke/Llama-2-7B-Chat-GPTQ`
- `TheBloke/Llama-2-13B-Chat-GPTQ`

## Usage Instructions

1. **Update the model name** in the script:
   ```python
   MODEL_NAME = "TheBloke/Llama-2-7B-Chat-AWQ"  # Replace with your model
   ```

2. **Adjust quantization type** if needed:
   ```python
   quantization="awq"  # Change to "gptq" if using GPTQ models
   ```

3. **Configure batch size** based on your GPU memory:
   - For 8GB GPU: `BATCH_SIZE = 8`
   - For 16GB GPU: `BATCH_SIZE = 16`
   - For 24GB+ GPU: `BATCH_SIZE = 32`

## Text Preprocessing Features

The script includes comprehensive text cleaning:

1. **Removes link_url placeholders**: Eliminates placeholder text
2. **URL removal**: Cleans remaining URLs and email addresses
3. **Punctuation normalization**: Reduces excessive punctuation
4. **Whitespace cleanup**: Normalizes spacing
5. **Character filtering**: Removes non-printable characters
6. **Validation checks**:
   - Minimum 2 words required
   - Minimum 10 characters
   - At least 30% alphabetic characters
   - Filters empty or meaningless content

## Additional Preprocessing Strategies

Consider implementing these additional strategies:

1. **Handle emojis**: Convert or remove emoji characters
2. **Normalize slang**: Create mappings for common Spanish slang
3. **Remove spam patterns**: Filter repetitive characters (aaaa, !!!!!)
4. **Language detection**: Ensure comments are in Spanish
5. **Profanity filtering**: Mark or handle offensive content
6. **Hashtag processing**: Extract hashtag content
7. **User mention removal**: Remove @username patterns

## Output Format

The script will generate:
- `base_obesidad_classified.csv`: Original data + sentiment classifications
- `base_obesidad_classified_analysis.json`: Detailed analysis results

## Running the Script

```bash
cd /home/ezequiel/ESTUDIO/trabajos_diplomatura/mentoria_2025
python sentiment_classification.py
```

## Sentiment Categories

- **POSITIVO**: Supportive, understanding, motivational content
- **NEGATIVO**: Discriminatory, mocking, hateful content  
- **NEUTRO**: Informational, descriptive, or emotionally neutral content
