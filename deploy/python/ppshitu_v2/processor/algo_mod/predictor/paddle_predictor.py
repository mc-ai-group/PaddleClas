import os
import platform
from paddle.inference import create_predictor
from paddle.inference import Config as PaddleConfig

from ...base_processor import BaseProcessor


class PaddlePredictor(BaseProcessor):
    def __init__(self, config):
        if config.get("use_fp16", False):
            assert config.get("use_tensorrt", False) is True

        inference_model_dir = config["inference_model_dir"]
        params_file = os.path.join(inference_model_dir, "inference.pdiparams")
        model_file = os.path.join(inference_model_dir, "inference.pdmodel")
        paddle_config = PaddleConfig(model_file, params_file)

        if config.get("use_gpu", False):
            paddle_config.enable_use_gpu(config.get("gpu_mem", 8000), 0)
        else:
            paddle_config.disable_gpu()
            if config.get("enable_mkldnn", False):
                # there is no set_mkldnn_cache_capatity() on macOS
                if platform.system() != "Darwin":
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    paddle_config.set_mkldnn_cache_capacity(10)
                paddle_config.enable_mkldnn()
        paddle_config.set_cpu_math_library_num_threads(
            config.get("cpu_num_threads", 10))

        if config.get("enable_profile", False):
            paddle_config.enable_profile()
        paddle_config.disable_glog_info()
        paddle_config.switch_ir_optim(config.get("ir_optim",
                                                 True))  # default true
        if config.get("use_tensorrt", True):
            paddle_config.enable_tensorrt_engine(
                precision_mode=PaddleConfig.Precision.Half
                if config.get("use_fp16", False) else
                PaddleConfig.Precision.Float32,
                max_batch_size=config.get("batch_size", 1),
                workspace_size=1 << 30,
                min_subgraph_size=30)

        paddle_config.enable_memory_optim()
        # use zero copy
        paddle_config.switch_use_feed_fetch_ops(False)
        self.predictor = create_predictor(paddle_config)

        if "input_names" in config and config["input_names"]:
            self.input_name_mapping = config["input_names"]
        else:
            self.input_name_mapping = []
        if "output_names" in config and config["output_names"]:
            self.output_name_mapping = config["output_names"]
        else:
            self.output_name_mapping = []

    def process(self, data):
        input_names = self.predictor.get_input_names()
        for input_name in input_names:
            input_tensor = self.predictor.get_input_handle(input_name)
            name = self.input_name_mapping[
                input_name] if input_name in self.input_name_mapping else input_name
            input_tensor.copy_from_cpu(data[name])
        self.predictor.run()

        output_data = {}
        output_names = self.predictor.get_output_names()
        for output_name in output_names:
            output = self.predictor.get_output_handle(output_name)
            name = self.output_name_mapping[
                output_name] if output_name in self.output_name_mapping else output_name
            output_data[name] = output.copy_to_cpu()
        data["pred"] = output_data
        return data
