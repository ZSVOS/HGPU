def get_dataset(args, split, image_transforms=None, target_transforms=None,
                augment=False, input_size=None, pre_train=False):

    from .dataloader import DataLoader as ChosenDataset

    dataset = ChosenDataset(args, split=split, transform=image_transforms,
                            target_transform=target_transforms, augment=augment,
                            input_size=input_size, pre_train=pre_train)
    return dataset
