- cuda
```python
python -c "import torch; [print(f'Device {i}: {torch.cuda.get_device_name(i)}\n') for i in range(torch.cuda.device_count())]"
```
