from diss_modules.reward import TextAlignmentReward
from torchvision import transforms
from integrations.dps.diss_sample_conditions import get_dataset, get_dataloader
import os
import shutil
import numpy as np

image_root = 'imagenet_data/val_set/'
text_root = 'captions/'


image_files = sorted(os.listdir(image_root))
text_files = sorted(os.listdir(text_root))


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

# Create a directory to store the top k images
os.makedirs('top_k_images', exist_ok=True)
os.makedirs('top_k_captions', exist_ok=True)

for j, idx in enumerate(top_k_ids):
    # Get the image file name
    image_file = image_files[idx]
    # Copy the image to the top k images directory
    shutil.copy(os.path.join(image_root, image_file), os.path.join('top_k_images', image_file))
    # Get the text file name
    text_file = text_files[idx]
    # Copy the text file to the top k captions directory
    shutil.copy(os.path.join(text_root, text_file), os.path.join('top_k_captions', text_file))
    print('clip score: ', sorted_clip_scores[j, 0], 'match: ', clip_scores[idx, 0])




# # Save the sorted clip_scores to a csv file
# np.savetxt('text_alignment_rewards.csv', sorted_clip_scores, delimiter=',', header='id,clip_score')











