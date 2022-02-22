from ghtools.predictor import predict

def test_pred():
    assert len(predict()) == 52
