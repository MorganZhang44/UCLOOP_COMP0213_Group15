import argparse
from Env.SimEnv import SimEnv
from ML.training import train_classifier_based_planner


def main():
    parser = argparse.ArgumentParser(description="Pybullet Grasping")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode (generator, training, testing)")
    
    # Generator
    Generator_parser = subparsers.add_parser("generator", help="Generate dataset")
    Generator_parser.add_argument("--gripper", type=str, required=True, help="Type of gripper.(2f,3f)")
    Generator_parser.add_argument("--object", type=str, required=True, help="Type of object.(cube, cylinder)")
    Generator_parser.add_argument("--num", type=int, required=True, help="Number of samples.")
    Generator_parser.add_argument("--output", type=str, required=True, help="Path to save dataset.")
    
    # Training
    Training_parser = subparsers.add_parser("training", help="Train classifier")
    Training_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset.")
    Training_parser.add_argument("--model", type=str, required=True, help="Path to save model.")
    Training_parser.add_argument("--test_size", type=float, required=False, help="Label column.")
    
    # Testing
    Testing_parser = subparsers.add_parser("testing", help="Test classifier")
    Testing_parser.add_argument("--gripper", type=str, required=True, help="Type of gripper.(2f,3f)")
    Testing_parser.add_argument("--object", type=str, required=True, help="Type of object.(cube, cylinder)")
    Testing_parser.add_argument("--model", type=str, required=True, help="Path to model.")
    Testing_parser.add_argument("--num", type=int, required=True, help="Number of samples.")
    
    
    args = parser.parse_args()
    
    if args.mode == "generator":
        env = SimEnv(robot=args.gripper, object=args.object)
        env.get_data(num=args.num,csv_path=args.output)
    elif args.mode == "training":
        if args.test_size is None:
            train_classifier_based_planner(dataset_path=args.dataset, model_output_path=args.model)
        else:
            train_classifier_based_planner(dataset_path=args.dataset, model_output_path=args.model, test_size=args.test_size)
    elif args.mode == "testing":
        env = SimEnv(robot=args.gripper, object=args.object)
        env.test(num=args.num,model_path=args.model)

if __name__ == "__main__":
    main()