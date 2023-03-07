"""言語モデルのベースクラス"""
import os
import pickle

from common.np import GPU, np
from common.util import to_cpu, to_gpu


class BaseModel:
    """言語モデルのベースクラス

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
    """
    def __init__(self):
        """コンストラクタ"""
        self.params = None
        self.grads = None

    def forward(self, *args):
        """順伝播

        Args:
            *args: 引数

        Raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError

    def backward(self, *args):
        """逆伝播

        Args:
            *args: 引数

        Raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError

    def save_params(self, file_name=None):
        """パラメータの保存

        Args:
            file_name (str, optional): ファイル名
        """
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        params = [param.astype(np.float16) for param in self.params]
        if GPU:
            params = [to_cpu(param) for param in self.params]

        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        """パラメータの読み込み

        Args:
            file_name (str, optional): ファイル名
        """
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        if "/" in file_name:
            file_name = file_name.replace("/", os.sep)

        if not os.path.exists(file_name):
            raise IOError("No file: " + file_name)

        with open(file_name, "rb") as f:
            params = pickle.load(f)

        params = [param.astype("f") for param in params]
        if GPU:
            params = [to_gpu(param) for param in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
