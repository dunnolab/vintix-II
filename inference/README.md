# How to Use Our Model

## Model Initialization
To initialize a model, you need three checkpoint files:

- `model.pth`: the model’s weights. 
- `config.json`: defines model parameters (e.g., `transformer_depth`, `transformer_heads`, `hidden_dim`) required to set up the model correctly. 
- `metadata.json`: contains task-related information (e.g., `group_name`, `reward_scale`).

Use the following code to initialize and load the checkpoint:

```python
PATH_TO_CHECKPOINT = "/path/to/checkpoint"
model = Vintix2()
model.load_model(PATH_TO_CHECKPOINT)
model.to(torch.device('cuda'))
model.eval()
```

## Preparing the Model
Before starting the sequential prediction, it’s important to prepare the model using the `reset_context` method. This method clears the current context and prepares a new one. You can also pass a prompt using this method.

An example of prediction without a prompt can be found in [online_inference.py](online_inference.py), and an example of prediction with a prompt can be found in [offline_inference.py](offline_inference.py).

## Action prediction
To predict the next action, use the `get_next_action` method. This method takes the current observation and the previous reward, then returns the current action. All required context updates are handled within this method.

## Advanced usage
You can use the `create_model_input` and `get_action` methods to write your own evaluation code:

1) `create_model_input` takes the sequences of observations, within-episode step numbers, previous actions, and previous rewards, then returns the appropriate input batch
2) `get_action` takes the input batch, and returns the action
