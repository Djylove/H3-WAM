import torch

from fastwam.models.h3wam.dense_future_value import DenseTemporalFutureValueModel


def test_dense_future_value_shapes_and_action_isolation():
    model=DenseTemporalFutureValueModel(h3_feature_dim=32,target_dim=16,hidden_dim=32,num_heads=4)
    state=torch.randn(3,8); hidden=torch.randn(3,1,32,32); actions=torch.randn(3,32,7,requires_grad=True)
    out=model(state,hidden,actions)
    assert out["future_h3"].shape==(3,16)
    assert out["future_state"].shape==(3,8)
    assert out["value"].shape==(3,)
    sum(x.square().mean() for x in out.values()).backward()
    assert actions.grad is None
    assert any(p.grad is not None for p in model.parameters())


def test_dense_future_value_is_action_sensitive():
    torch.manual_seed(7)
    model=DenseTemporalFutureValueModel(h3_feature_dim=32,target_dim=16,hidden_dim=32,num_heads=4).eval()
    state=torch.randn(1,8); hidden=torch.randn(1,32,32); a=torch.zeros(1,32,7); b=a.clone(); b[:,:,0]=1
    with torch.inference_mode(): x=model(state,hidden,a); y=model(state,hidden,b)
    assert not torch.equal(x["value"],y["value"])
    assert not torch.equal(x["future_h3"],y["future_h3"])
