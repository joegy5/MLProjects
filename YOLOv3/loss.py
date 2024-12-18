import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOv3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # for box predictions
        self.bce = nn.BCEWithLogitsLoss() # combines sigmoid layer and Binary cross entropy loss into one single class
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much each type of loss should be valued
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    # Compute loss for one scale --> call this function 3 times for the 3 scales
    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1 # vectorized implementation makes it binary tensor (true or false)
        noobj = target[..., 0] == 0

        #----------NO OBJECT LOSS----------
        # only select the ones that have no object
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))

        #----------OBJECT LOSS-------------
        # initially, the anchors will have shape 3 x 2 (3 anchors, height and width for each anchor)
        # new shape still has total space of 6, just has more dimensions 
        anchors = anchors.reshape(1, 3, 1, 1, 2) # remember that anchors contain values p_w & p_h as described in the paper --> reshape to be able to muliply with torch.exp of the predictions (using broadcasting)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5])*anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse((predictions[..., 0:1][obj]), (ious*target[..., 0:1][obj])) # p_o shouldn't just be 1 if there is object, but should be iou if there is object there, otherwise 0 like usual

        #----------BOX COORDINATE LOSS-----
        # invert the equations in the paper and then compute loss from that, for better gradient flow
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # x, y to be between 0 and 1 
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors) # log base e 
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        #----------CLASS LOSS--------------
        class_loss = self.entropy(predictions[..., 5:][obj], target[..., 5][obj].long())


        return self.lambda_box * box_loss + self.lambda_obj * object_loss + self.lambda_noobj * no_object_loss + self.lambda_class * class_loss
