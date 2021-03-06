from datasets.vehicle_attributes import VehicleAttributes
from datasets.human_attributes import HumanAttributes


def get_training_set(opt, transform):
    if opt.dataset == 'vehicle_attributes':
        training_data = VehicleAttributes(
            opt.root_dir,
            opt.train_path,
            opt.num_classes,
            transform = transform)
    if opt.dataset == 'human_attributes':
        training_data = HumanAttributes(
            opt.train_path,
            opt.num_classes,
            transform = transform)
    return training_data


def get_validation_set(opt, transform):
    if opt.dataset == 'vehicle_attributes':
        validation_data = VehicleAttributes(
            opt.root_dir,
            opt.val_path,
            opt.num_classes,
            transform=transform)
    if opt.dataset == 'human_attributes':
        validation_data = HumanAttributes(
            opt.val_path,
            opt.num_classes,
            transform = transform)
    return validation_data