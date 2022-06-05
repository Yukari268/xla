import argparse
from modules.converter import getImageFromVideo
from modules.trainer import Controller

if __name__ == "__main__":
    #Define Parser and Subparser
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command') #store all subparser to a propeties named command

    #vidtopics command
    vidtopics = subparser.add_parser('vidtopics')
    train = subparser.add_parser('train')
    test = subparser.add_parser('test')

    #vidtopics arguments
    vidtopics.add_argument('videourl', type=str)

    #test arguments
    test.add_argument('pictureUrl', type=str)
    test.add_argument('--algorithm','--al', type=str, default='knn', const='knn', nargs="?")
    args = parser.parse_args()
    controller = Controller()

    if args.command == 'vidtopics':
        print("Trying to convert video ",args.videourl)
        getImageFromVideo(args.videourl)
    elif args.command == 'train':
        print("Training...")
        trainer = controller.getTrainer()
        embedded = trainer.preProcess()
        trainer.train(embedded=embedded)
        trainer.save()
        print("Trained")
    elif args.command == 'test':
        tester = controller.getTester()
        tester.loadModel("./knn.pkl") if (args.algorithm == 'knn') else tester.loadModel("./svc.pkl")
        tester.indentifyImage(imgUrl= args.pictureUrl, mydict=['BanNinh01','BanThanh01','CongDanh1','GiaQuang','HoaiBao','HoangLong1','HoangNhat1','KimHung1','MinhHien1','MinhHieu1','MinhHoa1','MinhNhat1','MyLinh01','TanTin1','ThanhLoi1','ThanhTung1','ThuyLinh1','TrongHuy1','XuanPhuc1'])