# LOSiamMask

```C++
//  yolov5 .torchscript.pt models
    parser.addArgument("-w","--weights",1,false);
//    yolo classes name
    parser.addArgument("-n","--names",1,true);
//  siammask model
    parser.addArgument("-m", "--modeldir", 1, false);
//  siammask config  config_vot.json
    parser.addArgument("-c", "--config", 1, false);
//  first image
    // parser.addArgument("-s","--source",1,true);
//  iou conf
    parser.addArgument("--iou",1,true);
    parser.addArgument("--conf",1,true);
//    image dir
    parser.addFinalArgument("target");
```


./LOSiamMask -w ../weights/yolov5s.torchscript.pt -m ../models/SiamMask_VOT -c ../config_vot.json ../images/tennis/
