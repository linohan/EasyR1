from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader



data_path = "/Users/ljm/Documents/Code/z_dev/dataset_collect/math12k/manual_corrected-20250311/"
dataset = load_dataset("parquet", data_dir=data_path, split="train")
print(len(dataset))
pass


train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

