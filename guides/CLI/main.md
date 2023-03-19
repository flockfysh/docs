## Build and upload to PyPI

To build the package: 
```shell
pip install build
python -m build
```

To upload the package: 
```shell
pip install twine
python -m twine upload --repository pypi dist/*
```


<!-- # ffysh

[Download example dataset (**DO NOT EXTRACT
**)](https://drive.google.com/file/d/1oEETaG6Ra_Ajq5j2awcEXdwmYbW4d2zg/view?usp=sharing) -->

# General use

Everything you need is stored in the ffysh import or in the ffysh CLI - you don't need to worry about its submodules.

## Create a project and log in to Flockfysh

1. Ensure your current working directory is correct.
2. From the terminal, run these two commands:
    ```shell
    ffysh init
    ffysh login
    ```
3. Log in to Flockfysh using the new browser window and click `Approve` to allow ffysh to use your account.

## Datasets

1. You'll first need to create a Dataset object first, using the 24-character ID of the dataset you want.
   If the dataset does not exist, an error will be thrown.

   ```python
   from ffysh import Dataset
   
   dataset = Dataset("some_dataset_id")
   ```

2. From there, you can create a stream object.
   ```python
   stream = dataset.create_stream()
   ```

Additional dataset attributes and methods:

```python
# Print dataset ID.
print(dataset.dataset_id)
```

## Streams

Streams are the basic unit of operation in ffysh.
Each stream is a lazy-loaded snapshot of a dataset, and it expands as much as the user needs.
Alternatively, you can use it as if it is a Python list.

1. First, store the ID of the stream for later use.

   ```python
   with open("stream_id.json", "w") as file:
       json.dump({"id": stream.stream_id}, file)
   ```

2. This allows the stream to reload from disk.

   ```python
   from ffysh import Stream
   
   with open("stream_id.json", "w") as file:
      stream = Stream.load(json.load(file)["id"]
   ```

   Note that each stream instance can only be created or loaded once, and if it is load twice, the old stream instance
   will be returned.

   ```python
   with open("stream_id.json", "w") as file:
      stream2 = Stream.load(json.load(file)["id"]
   
   print(stream is stream2) # True
   ```

3. Stream methods:
   ```python
   from ffysh import Stream
   
   with open("stream_id.json", "w") as file:
       stream = Stream.load(json.load(file)["id"]
   
   # Iteration
   for asset in stream:
       print(asset)
   
   # Stream indexing and slicing
   print(stream[0:5:2])
   
   # Loads the next 5 items that has not been loaded yet 
   # into the cache, and return them. If the stream ends, 
   # an empty list is returned, and any more calls will raise 
   # a StopIteration exception.
   stream.next_assets(5)
   
   # Expands the stream until the stream ends or is at least 100 
   # items long by loading more items from the cache.
   stream.expand(100)
   
   # Expands the stream until the end.
   stream.expand()
   
   # Convert the entire stream to a PyTorch zip.
   stream.to_pytorch("./dataset.zip", confidence_level=0.6)
   ```

## Limitations
1. You must not use a stream in 2 Python programs at a time. If you want to, create a second 
stream or quit the first program. Although you still have to re-download labels, images will be 
cached in a separate directory, so that multiple streams can access the same asset.
2. Even if assets and their labels are deleted remotely, they will stay intact locally.
We'll introduce a way to discard unused streams and purge redundant images in the future.
