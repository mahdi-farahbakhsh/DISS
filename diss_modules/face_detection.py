import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

class FaceRecognition(nn.Module):
    def __init__(self, fr_crop=False, mtcnn_face=False, norm_order=2):
        super().__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        print(self.resnet)
        self.mtcnn = MTCNN(device='cuda').eval()
        self.crop = fr_crop
        self.output_size = 160
        self.mtcnn_face = mtcnn_face
        self.norm_order = norm_order

    def extract_face(self, imgs, batch_boxes, mtcnn_face=False):
        image_size = imgs.shape[-1]
        faces = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            if not mtcnn_face:
                box = [48, 48, 208, 208]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            elif batch_boxes[i] is not None:
                box = batch_boxes[i][0]
                margin = [
                    self.mtcnn.margin * (box[2] - box[0]) / (self.output_size - self.mtcnn.margin),
                    self.mtcnn.margin * (box[3] - box[1]) / (self.output_size - self.mtcnn.margin),
                ]

                box = [
                    int(max(box[0] - margin[0] / 2, 0)),
                    int(max(box[1] - margin[1] / 2, 0)),
                    int(min(box[2] + margin[0] / 2, image_size)),
                    int(min(box[3] + margin[1] / 2, image_size)),
                ]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            else:
                # crop_face = img[None, :, :, :]
                return None

            faces.append(F.interpolate(crop_face, size=self.output_size, mode='bicubic'))
        new_faces = torch.cat(faces)

        return (new_faces - 127.5) / 128.0

    def get_faces(self, x, mtcnn_face=False):
        img = (x + 1.0) * 0.5 * 255.0
        img = img.permute(0, 2, 3, 1)
        with torch.no_grad():
            batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
            # Select faces
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.mtcnn.selection_method
            )

        img = img.permute(0, 3, 1, 2)
        faces = self.extract_face(img, batch_boxes, mtcnn_face)
        return faces

    def forward(self, x, return_faces=False, mtcnn_face=None):
        x = TF.resize(x, (256, 256), interpolation=TF.InterpolationMode.BICUBIC)

        if mtcnn_face is None:
            mtcnn_face = self.mtcnn_face

        faces = self.get_faces(x, mtcnn_face=mtcnn_face)
        if faces is None:
            return faces

        if not self.crop:
            out = self.resnet(x)
        else:
            out = self.resnet(faces)

        if return_faces:
            return out, faces
        else:
            return out

    def cuda(self):
        self.resnet = self.resnet.cuda()
        self.mtcnn = self.mtcnn.cuda()
        return self
    

    def compute_loss(self, x_tilde, x):
        """
        Computes the face similarity loss between two images.
        
        Parameters:
            x (torch.Tensor): The first image [-1, 1] tensor.
            x_tilde (torch.Tensor): The second image [-1, 1] tensor.
        
        Returns:
            torch.Tensor: The computed face similarity loss.
        """
        
        # Compute the embeddings for both images
        with torch.no_grad():
            z_tilde = self(x_tilde)

        print('z_tilde shape:', z_tilde.shape)

        z = torch.zeros_like(z_tilde)

        for i, x_i in enumerate(x):
            with torch.no_grad():
                z_i = self(x_i.unsqueeze(0))

            if z_i is not None:
                z[i] = z_i.squeeze(0)
            else:
                z[i] = torch.zeros_like(z_tilde[i])

        diff = z - z_tilde
        # Compute error norm based on the specified norm order
       
        loss = torch.norm(diff, p=self.norm_order, dim=1)
            
        return loss
    

    def compute_loss_and_gradient(self, x, x_tilde):
        """
        Computes the face similarity loss between two images.
        
        Parameters:
            x (torch.Tensor): The first image [-1, 1] tensor.
            x_tilde (torch.Tensor): The second image [-1, 1] tensor.
        
        Returns:
            torch.Tensor: The computed face similarity loss.
        """
        
        # Compute the embeddings for both images
        with torch.no_grad():
            z_tilde = self(x_tilde)

        z = torch.zeros_like(z_tilde)

        x.requires_grad_()
        with torch.enable_grad():
            
            for i, x_i in enumerate(x):
                z_i = self(x_i.unsqueeze(0))

                if z_i is not None:
                    z[i] = z_i.squeeze(0)
                else:
                    z[i] = torch.zeros_like(z_tilde[i])

            diff = z - z_tilde
            # Compute error norm based on the specified norm order
           
            loss = torch.norm(diff, p=self.norm_order, dim=1) ** self.norm_order

            print('Fixed the power of the norm depend on norm_order')
                
            norm_grad = torch.autograd.grad(loss.sum(), x)[0]
        
        x.requires_grad_(False)
        return loss, norm_grad

        

if __name__ == "__main__":
    # Example usage
    face_recognition = FaceRecognition(mtcnn_face=True)
    face_recognition.cuda()

    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    # Load a sample image
    ref_img_1 = Image.open('../../ood_images/0000.png').convert('RGB')
    guid_img_1 = Image.open('../../ood_images/0001.png').convert('RGB')

    ref_tensor_1 = transform(ref_img_1).unsqueeze(0).cuda()
    guid_tensor_1 = transform(guid_img_1).unsqueeze(0).cuda()


    # Load a sample image
    ref_img_2 = Image.open('../../ood_images/1000.png').convert('RGB')
    guid_img_2 = Image.open('../../ood_images/1001.png').convert('RGB')

    ref_tensor_2 = transform(ref_img_2).unsqueeze(0).cuda()
    guid_tensor_2 = transform(guid_img_2).unsqueeze(0).cuda()

    ref_tensor_1 = torch.randn_like(ref_tensor_2).cuda()

    ref = torch.cat([ref_tensor_1, ref_tensor_2], dim=0)
    guid = torch.cat([guid_tensor_1, guid_tensor_2], dim=0)
    print("Reference Shape:", ref.shape)
    print("Guid Shape:", guid.shape)

    # ref_emb = face_recognition(ref)
    # guid_emb = face_recognition(guid)

    # diff = ref_emb - guid_emb
    # print("Reference Embedding:", diff.shape)

    # # Compute L2 norm
    # l2_norm = torch.norm(diff, dim=1)
    # print("L2 Norm:", l2_norm)

    # Compute face similarity loss with L2 norm
    loss_l2, norm_l2_grad = face_recognition.compute_loss_and_gradient(ref, guid, norm_order=2)
    print("Face Similarity Loss (L2):", loss_l2)
    print("Gradient Shape (L2):", norm_l2_grad.shape)
    # Compute face similarity loss with L1 norm
    loss_l1, norm_l1_grad = face_recognition.compute_loss_and_gradient(ref, guid, norm_order=1)
    print("Gradient Shape (L1):", norm_l1_grad.shape)
    print("Face Similarity Loss (L1):", loss_l1)