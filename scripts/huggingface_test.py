from datasets import Dataset
data = [[1, 2],[3, 4]]
ds = Dataset.from_dict({"data": data})
ds = ds.with_format("torch")
print(ds[0])
print(ds[:2])
