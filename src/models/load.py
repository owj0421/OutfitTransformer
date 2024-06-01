from src.datasets.processor import FashionInputProcessor, FashionImageProcessor
from transformers import CLIPImageProcessor, CLIPTokenizer, AutoTokenizer
from src.models.embedder import CLIPEmbeddingModel, OutfitTransformerEmbeddingModel
from src.models.recommender import RecommendationModel
from src.utils.utils import load_model_from_checkpoint

def load_model(args):
    model_type = 'Clip Embedding' if args.use_clip_embedding else 'OutfitTransformer Embedding'
    print(f'Use {model_type} for recommendation.')
    if args.use_clip_embedding:
        image_processor = CLIPImageProcessor.from_pretrained(args.clip_huggingface)
        text_tokenizer = CLIPTokenizer.from_pretrained(args.clip_huggingface)
    else:
        image_processor = FashionImageProcessor()
        text_tokenizer = AutoTokenizer.from_pretrained(args.huggingface)

    input_processor = FashionInputProcessor(
        categories=args.categories,
        use_image=args.use_image,
        image_processor=image_processor, 
        use_text=args.use_text,
        text_tokenizer=text_tokenizer, 
        text_max_length=args.text_max_length, 
        text_padding='max_length', 
        text_truncation=True, 
        outfit_max_length=args.outfit_max_length
        )
    
    if args.use_clip_embedding:
        embedding_model = CLIPEmbeddingModel(
            input_processor=input_processor,
            hidden=args.hidden,
            huggingface=args.clip_huggingface,
            fp16=True,
            linear_probing=True,
            normalize=True
            )
    else:
        embedding_model = OutfitTransformerEmbeddingModel(
            input_processor=input_processor,
            hidden=args.hidden,
            huggingface=args.huggingface,
            fp16=True,
            linear_probing=True,
            normalize=True
            )

    recommendation_model = RecommendationModel(
        embedding_model=embedding_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        normalize=True
        )
    
    if args.load_model:
        load_model_from_checkpoint(recommendation_model, args.model_path)
    
    return recommendation_model, input_processor