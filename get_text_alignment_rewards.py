from diss_modules.reward import TextAlignmentReward
from torchvision import transforms
from integrations.dps.diss_sample_conditions import get_dataset, get_dataloader
import os
import shutil
import numpy as np

image_root = './imagenet_test_data/images/'
text_root = './imagenet_test_data/captions/'


image_files = sorted(os.listdir(image_root))

data_config = {
    'name': 'imagenet',
    'root': image_root,
}


transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)


reward = TextAlignmentReward(data_path=text_root, pretrained_model='ViT-B/32', resolution=256, device='cuda:0', scale=1)

clip_scores = np.zeros((len(loader), 2))

# # Create a csv file to store the rewards
# with open('text_alignment_rewards.csv', 'w') as f:
#     f.write('id,clip_score\n')


for i, ref_img in enumerate(loader):
    reward.set_side_info(i) 
    clip_score = reward.get_reward(ref_img)
    clip_scores[i, 0] = clip_score.item()
    clip_scores[i, 1] = i
    
    # print(f'{i}, {clip_score}')
    # with open('text_alignment_rewards.csv', 'a') as f:
    #     f.write(f'{i}, {clip_score.item()}\n')


# Sort the clip_scores by the first column into a new array in descending order
sorted_clip_scores = clip_scores[(clip_scores[:, 0].argsort(axis=0)[::-1]).astype(int)]

top_k = 50

top_k_ids = sorted_clip_scores[:top_k, 1].astype(int)

print('top k ids: ', top_k_ids)


sorted_image_path = 'imagenet_test_data/ordered_images'
sorted_caption_path = 'imagenet_test_data/ordered_captions'

# Create a directory to store the top k images
os.makedirs(sorted_image_path, exist_ok=True)
os.makedirs(sorted_caption_path, exist_ok=True)

for j, idx in enumerate(top_k_ids):
    # Get the image file name
    image_file = image_files[idx]
    # the image to the top k images directory
    shutil.copy(os.path.join(image_root, image_file), os.path.join(sorted_image_path, f"{j:05d}_{image_file}"))
    # Get the text file name
    text_file = image_files[idx].replace('.jpg', '.txt')
    # Copy the text file to the top k captions directory
    shutil.copy(os.path.join(text_root, text_file), os.path.join(sorted_caption_path, f"{j:05d}_{text_file}"))
    print('clip score: ', sorted_clip_scores[j, 0], 'match: ', clip_scores[idx, 0])




# # Save the sorted clip_scores to a csv file
# np.savetxt('text_alignment_rewards.csv', sorted_clip_scores, delimiter=',', header='id,clip_score')











