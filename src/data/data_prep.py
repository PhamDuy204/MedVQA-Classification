from PIL import Image
from torch.utils.data import Dataset
import glob
import os
import json
def map_sample2img(samples,imge_fullpaths):
    img_name = 'img_name' if samples[0].get('img_name','image_name') != 'image_name' else 'image_name'
    cnt = samples[0][img_name].strip().strip('.').strip('/').count('/')
    
    return {
        '/'.join(img_path.split('/')[-(cnt+1):]): img_path for img_path in imge_fullpaths
    }
    

class PreDataset(Dataset):
    def __init__(self, path_dataset,type_split='train'):
        img_paths = glob.glob(
                            os.path.join(path_dataset, "**", "*.png"),
                            recursive=True
                        )+glob.glob(
                            os.path.join(path_dataset, "**", "*.jpg"),
                            recursive=True
                        )
        self.samples = list(filter(lambda x : x.get("q_lang","en")=="en",
            json.load(open(glob.glob(
                                    os.path.join(path_dataset, "**", f"{type_split}*.json"),
                                    recursive=True
                                )[0],'r'))))
        self.map_imgName = map_sample2img(self.samples,img_paths)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        sample_i = self.samples[index]
        img_name = 'img_name' if sample_i.get('img_name','image_name') != 'image_name' else 'image_name'
        img_path = self.map_imgName[sample_i[img_name].strip().strip('.').strip('/')]
        question = sample_i['question']
        answer = sample_i['answer']
        answer_type = sample_i['answer_type']
        return {
            'image':Image.open(img_path),
            'question': question,
            'answer':answer,
            'answer_type':answer_type
        }
