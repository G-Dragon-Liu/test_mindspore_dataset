import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as v_c_trans
from multiprocessing import cpu_count
import numpy as np
from typing import cast
import os


dataset_dir = "./data"
wrong_dataset_dir = "./invalid"


'''
    3.1.1 输入
'''

'''
    3.1.2 入参
'''

'''
TEST_SUMMARY: test EMnistDataset with no para
'''

def test_no_para():
    with pytest.raises(TypeError, match="missing a required argument: 'dataset_dir'"):
        dataset = ds.EMnistDataset()

'''
TEST_SUMMARY: test EMnistDataset with more para
'''

def test_more_para():
    num_samples = 2
    num_parallel_workers = cpu_count()
    shuffle = True
    num_shards = 3
    shard_id = 2
    more_para = None
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'more_para'"):
        dataset = ds.EMnistDataset(dataset_dir, usage="train", usage="all", num_samples=num_samples, num_parallel_workers=num_parallel_workers,
                                shuffle = shuffle, num_shards=num_shards, shard_id=shard_id, more_para=more_para)

'''
TEST_SUMMARY: test EMnistDataset with invalid type of dataset_dir (str)
'''

def test_invalid_type_dataset_dir():
    with pytest.raises(TypeError, match="is not of type \[<class 'str'>\], but got <class 'list'>"):
        dataset = ds.EMnistDataset([dataset_dir])
    with pytest.raises(TypeError, match="is not of type \[<class 'str'>\], but got <class 'int'>"):
        dataset = ds.EMnistDataset(1)

'''
TEST_SUMMARY: test EMnistDataset with nonexist dataset_dir
'''

def test_nonexist_dataset_dir():
    with pytest.raises(ValueError, match="does not exist or is not a directory or permission denied!"):
        dataset = ds.EMnistDataset("")
    with pytest.raises(ValueError, match="does not exist or is not a directory or permission denied!"):
        dataset = ds.EMnistDataset("12no34exist56path78")

'''
TEST_SUMMARY: test EMnistDataset with wrong dataset_dir
'''

def test_wrong_dataset_dir():
    with pytest.raises(ValueError, match="does not exist or is not a directory or permission denied!"):
        dataset = ds.EMnistDataset(wrong_dataset_dir)

'''
TEST_SUMMARY: test EMnistDataset with invalid type of usage (str)
'''

def test_invalid_type_usage():
    with pytest.raises(TypeError, match="is not of type \[<class 'str'>\], but got <class 'list'>"):
        dataset = ds.USPSDataset(dataset_dir, usage=["all"])
    with pytest.raises(TypeError, match="is not of type \[<class 'str'>\], but got <class 'int'>"):
        dataset = ds.USPSDataset(dataset_dir, usage=1)

'''
TEST_SUMMARY: test USPSDataset with invalid usage
'''

def test_invalid_usage():
    with pytest.raises(ValueError, match="Input usage is not within the valid set of"):
        dataset = ds.USPSDataset(dataset_dir, usage="invalid")

'''
TEST_SUMMARY: test USPSDataset with invalid type of num_parallel_workers (int)
'''

def test_invalid_type_num_parallel_workers():
    with pytest.raises(TypeError, match="is not of type \[<class 'int'>\], but got <class 'str'>."):
        dataset = ds.USPSDataset(dataset_dir, usage="all", num_parallel_workers="")
    with pytest.raises(TypeError, match="is not of type \[<class 'int'>\], but got <class 'list'>."):
        dataset = ds.USPSDataset(dataset_dir, usage="all", num_parallel_workers=[1])

'''
TEST_SUMMARY: test USPSDataset with num_parallel_workers equal to cpu_count()
'''

def test_num_parallel_workers_equal_to_cpu_count():
    num_parallel_workers = cpu_count()
    dataset = ds.USPSDataset(dataset_dir, usage="all", num_parallel_workers=num_parallel_workers)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 9298

'''
TEST_SUMMARY: test USPSDataset with num_parallel_workers less than cpu_count()
'''

def test_num_parallel_workers_less_than_cpu_count():
    num_parallel_workers = cpu_count() - 1
    dataset = ds.USPSDataset(dataset_dir, usage="all", num_parallel_workers=num_parallel_workers)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 9298

'''
TEST_SUMMARY: test USPSDataset with num_parallel_workers equal to 1
'''

def test_num_parallel_workers_equal_to_1():
    num_parallel_workers = 1
    dataset = ds.USPSDataset(dataset_dir, usage="all", num_parallel_workers=num_parallel_workers)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 9298

'''
TEST_SUMMARY: test USPSDataset with invalid num_parallel_workers
'''

def test_invalid_num_parallel_workers():
    num_parallel_workers = cpu_count() + 1
    with pytest.raises(ValueError, match="exceeds the boundary between"):
        dataset = ds.USPSDataset(dataset_dir, usage="all", num_parallel_workers=num_parallel_workers)
    with pytest.raises(ValueError, match="exceeds the boundary between"):
        dataset = ds.USPSDataset(dataset_dir, usage="all", num_parallel_workers=-1)

'''
TEST_SUMMARY: test USPSDataset with invalid type of num_samples (int)
'''

def test_invalid_type_num_samples():
    with pytest.raises(TypeError, match="is not of type \[<class 'int'>\], but got <class 'str'>."):
        dataset = ds.USPSDataset(dataset_dir, usage="all", num_samples="")
    with pytest.raises(TypeError, match="is not of type \[<class 'int'>\], but got <class 'list'>."):
        dataset = ds.USPSDataset(dataset_dir, usage="all", num_samples=[100])


'''
TEST_SUMMARY: test USPSDataset with num_samples
'''

def test_num_samples_10():
    num_samples = 10
    dataset = ds.USPSDataset(dataset_dir, usage="all", num_samples=num_samples)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        num_iter += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert num_iter == 10

'''
TEST_SUMMARY: test USPSDataset with invalid num_samples
'''

def test_invalid_num_samples():
    with pytest.raises(ValueError, match="exceeds the boundary between"):
        dataset = ds.USPSDataset(dataset_dir, usage="all", num_samples=-1)

'''
TEST_SUMMARY: test USPSDataset with invalid type of shuffle (bool)
'''

def test_invalid_type_shuffle():
    with pytest.raises(TypeError, match="shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or 'Shuffle.FILES' or 'Shuffle.INFILE'."):
        dataset = ds.USPSDataset(dataset_dir, shuffle="")
    with pytest.raises(TypeError, match="shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or 'Shuffle.FILES' or 'Shuffle.INFILE'."):
        dataset = ds.USPSDataset(dataset_dir, shuffle=1)

'''
TEST_SUMMARY: test USPSDataset with num_shards is 19 shard_id is 0
'''

def test_num_shards_19():
    num_shards = 19
    shard_id = 0
    dataset = ds.USPSDataset(dataset_dir, usage="all", num_shards=num_shards, shard_id=shard_id)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    # 9298 / 19 = 490
    assert num_iter == 490

'''
TEST_SUMMARY: test USPSDataset with num_shards is 1
'''

def test_num_shards_1():
    num_shards = 1
    shard_id = 0
    dataset = ds.USPSDataset(dataset_dir, usage="all", num_shards=num_shards, shard_id=shard_id)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 9298

'''
TEST_SUMMARY: test USPSDataset exception when num_shards is set but shard_id is None
'''

def test_require_shard_id():
    with pytest.raises(RuntimeError, match="num_shards is specified and currently requires shard_id as well"):
        ds.USPSDataset(dataset_dir, usage="all", num_shards=10)

'''
TEST_SUMMARY: test USPSDataset exception when shard_id is set but num_shards is None
'''

def test_require_num_shards():
    with pytest.raises(RuntimeError, match="shard_id is specified but num_shards is not"):
        ds.USPSDataset(dataset_dir, usage="all", shard_id=0)

'''
TEST_SUMMARY: test USPSDataset exception when shard_id is not within the required interval
'''

def test_invalid_shard_id():
    with pytest.raises(ValueError, match="Input shard_id is not within the required interval"):
        ds.USPSDataset(dataset_dir, usage="all", num_shards=10, shard_id=-1)
    with pytest.raises(ValueError, match="Input shard_id is not within the required interval"):
        ds.USPSDataset(dataset_dir, usage="all", num_shards=10, shard_id=10)
    with pytest.raises(ValueError, match="Input shard_id is not within the required interval"):
        ds.USPSDataset(dataset_dir, usage="all", num_shards=6, shard_id=10)


'''
TEST_SUMMARY: test USPSDataset with default para
'''

def test_default_para():
    dataset_1 = ds.USPSDataset(dataset_dir, shuffle=False)
    dataset_2 = ds.USPSDataset(dataset_dir, usage="all", shuffle=False)
    num_iter = 0
    for item1, item2 in zip(dataset_1.create_dict_iterator(output_numpy=True), dataset_2.create_dict_iterator(output_numpy=True)):
        np.testing.assert_array_equal(item1["image"], item2["image"])
        np.testing.assert_array_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == 9298

'''
    3.1.3 功能用例-数据集加载算子 : 对标数据算子
'''

'''
TEST_SUMMARY: test USPSDataset content
'''

def test_content():
    def load_usps(path, usage):
        """
        load USPS data
        """
        assert usage in ["train", "test"]
        if usage == "train":
            data_path = os.path.realpath(os.path.join(path, "usps"))
        elif usage == "test":
            data_path = os.path.realpath(os.path.join(path, "usps.t"))

        with open(data_path, 'r') as f:
            raw_data = [line.split() for line in f.readlines()]
            tmp_list = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
            images = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16, 1))
            images = ((cast(np.ndarray, images) + 1) / 2 * 255).astype(dtype=np.uint8)
            labels = [int(d[0]) - 1 for d in raw_data]
        return images, labels

    dataset = ds.USPSDataset(dataset_dir, usage="train", shuffle=False)
    images, labels = load_usps(dataset_dir, "train")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        for m in range(16):
            for n in range(16):
                assert (data["image"][m, n, 0] != 0 or images[i][m, n, 0] != 255) and \
                        (data["image"][m, n, 0] != 255 or images[i][m, n, 0] != 0)
                assert (data["image"][m, n, 0] == images[i][m, n, 0]) or\
                        (data["image"][m, n, 0] == images[i][m, n, 0] + 1) or\
                        (data["image"][m, n, 0] + 1 == images[i][m, n, 0])
        num_iter += 1
    assert num_iter == 7291

    dataset = ds.USPSDataset(dataset_dir, usage="test", shuffle=False)
    images, labels = load_usps(dataset_dir, "test")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        for m in range(16):
            for n in range(16):
                assert (data["image"][m, n, 0] != 0 or images[i][m, n, 0] != 255) and \
                        (data["image"][m, n, 0] != 255 or images[i][m, n, 0] != 0)
                assert (data["image"][m, n, 0] == images[i][m, n, 0]) or\
                        (data["image"][m, n, 0] == images[i][m, n, 0] + 1) or\
                        (data["image"][m, n, 0] + 1 == images[i][m, n, 0])
        num_iter += 1
    assert num_iter == 2007

'''
    3.1.3 功能用例-数据集加载算子 : basic_ops
    所有数据集basic_ops验证算子 （batch / shuffle / map）
'''

'''
TEST_SUMMARY: test USPSDataset with shuffle op
'''

def test_shuffle_op():
    buffer_size = 6
    dataset = ds.USPSDataset(dataset_dir, usage="all")
    ds.config.set_seed(58)
    dataset = dataset.shuffle(buffer_size)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        i += 1
    assert i == 9298

'''
TEST_SUMMARY: test USPSDataset with map op
'''

def test_map_op():
    input_columns = ["image"]
    image1, image2 = [], []
    dataset = ds.USPSDataset(dataset_dir, usage="all")
    for data in dataset.create_dict_iterator(output_numpy=True):
        image1.extend(data['image'])
    operations = [(lambda x: x + x)]
    dataset = dataset.map(input_columns=input_columns, operations=operations)
    for data in dataset.create_dict_iterator(output_numpy=True):
        image2.extend(data['image'])
    assert len(image1) == len(image2)

'''
TEST_SUMMARY: test USPSDataset with batch op
'''

def test_batch_op():
    dataset = ds.USPSDataset(dataset_dir, usage="all")
    dataset = dataset.batch(batch_size=19)

    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        num_iter += 1
    assert num_iter == 490


'''
    3.1.3 功能用例-数据集加载算子 : basic_ops
    mappable数据集basic_ops验证（use_sampler/split）
'''

'''
    3.1.4 pipeline
'''
def test_decode_op():
    dataset = ds.USPSDataset(dataset_dir, "all")
    with pytest.raises(RuntimeError, match=" Decode: invalid input shape, only support 1D input"):
        decode_op = v_c_trans.Decode()
        dataset = dataset.map(input_columns=["image"], operations=decode_op)
        for data in dataset.create_dict_iterator(output_numpy=True):
            image = data["image"]
            label = data["label"]

'''
    验证继承关系
'''

'''
TEST_SUMMARY: test USPSDataset is instance of SourceDataset
'''

def test_isinstance():
    isinstance(ds.USPSDataset, ds.SourceDataset)
