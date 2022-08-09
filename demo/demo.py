from demo_utils import Visualizer
import torch
class Args:
    def __init__(self, dataset, config, output_dir, checkpoint, device, split='test', evaluate=True, load_pretrained=False) -> None:
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.evaluate = evaluate
        self.load_pretrained = load_pretrained
        self.device = device
        self.split = split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")
args = Args(dataset='re', config='./itr_coco/config.yaml', output_dir='./',\
     checkpoint='../checkpoints/itr_coco/checkpoint_9.pth', device=device)

visualizer = Visualizer(args)

visualizer.index_images()
visualizer.show_results("A tall man in black dress")