from transformers import TrainingArguments, Trainer
from peft import get_model_for_kbit_training, get_peft_model, LoraConfig, TaskType

# Lora
def train_model(model, train_data, valid_data, training_args, cfg):
    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_models=cfg.model.lora_target_modules,
    )

    # 
    model = get_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_datasets=train_data,
        eval_data=valid_data,
    )

    # train
    trainer.train()

    # model save
    model.save_pretrained(cfg.train.output_dir)