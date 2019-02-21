(* ::Package:: *)

Clear[params]
<<NeuralNetworks`
SetDirectory@NotebookDirectory[];
params=Import@"karras2019stylegan-ffhq-1024x1024.wxf";


embedFC[i_]:=LinearLayer[512,
	"Weights"->Normal@params["G_mapping/Dense"<>ToString[i]<>"/weight"]/1600,
	"Biases"->Normal@params["G_mapping/Dense"<>ToString[i]<>"/bias"]/100
]


getCN[name_,wScale_,p_:1]:=GeneralUtilities`Scope[
	weight=Normal@params["G_synthesis/"<>name<>"/weight"];
	trans=TransposeLayer[{1<->4,2<->3,3<->4}];
	ConvolutionLayer[
		"Weights"->trans@weight/wScale,
		"Biases"->None,
		"PaddingSize"->p
	]
]
styleModify[name_,channal_,size_,wScale_]:=GeneralUtilities`Scope[
	weight=params["G_synthesis/"<>name<>"/StyleMod/weight"];
	bias=params["G_synthesis/"<>name<>"/StyleMod/bias"];
	style=StyleModifyLayer[channal,Length@bias,size,
		"Weights"->Transpose[Normal@weight/wScale,1<->2],
		"Biases"->bias
	]
]
styleModifyNative[name_,channel_,wScale_]:=GeneralUtilities`Scope[
	weight=params["G_synthesis/"<>name<>"/StyleMod/weight"];
	bias=params["G_synthesis/"<>name<>"/StyleMod/bias"];
	linear=LinearLayer[1024,
		"Weights"->Transpose[Normal@weight/wScale,1<->2],
		"Biases"->Length@bias
	];
	layers={
		"Affine"->linear,
		"Slice"->ReshapeLayer[{2,channel}],
		"Head"->PartLayer[1],
		"Tail"->PartLayer[-1],
		"Shift"->ElementwiseLayer[#+1&],
		"Mix"->ThreadingLayer[Times],
		"Fuse"->ThreadingLayer[Plus]
	};
	path={
		NetPort["Style"]->"Affine"->"Slice"->{"Head","Tail"->"Shift"},
		{"Shift", NetPort["Input"]}->"Mix",
		{"Head","Mix"}->"Fuse"->NetPort["Output"]
	};
	NetGraph[layers,path]
]
getBlur[channal_]:=GeneralUtilities`Scope[
	blur={{0.0625,0.125,0.0625},{0.125,0.25,0.125},{0.0625,0.125,0.0625}};
	DepthwiseConvolution[
		"Weights"->Transpose[ConstantArray[blur,{1,channal}],1<->2],
		"Biases"->None,"PaddingSize"->1
	]
]
styleNoise[name_,channal_,size_]:=StyleNoiseLayer[
	channal,size,
	"Weights"->params["G_synthesis/"<>name<>"/Noise/weight"],
	"Biases"->params["G_synthesis/"<>name<>"/bias"]
]


getBlockIn[i_,c_,s_,{w1_,w2_}]:=GeneralUtilities`Scope[
	name=StringRiffle[{i,"x",i},""];
	norm=InstanceNormalizationLayer["Epsilon"->10^-8,"Input"->{c,s,s}];
	layers={
		"Affine"->getCN[name<>"/Conv0_up",w1],
		"Blur"->getBlur[c],
		"Noise"->styleNoise[name<>"/Conv0_up",c,s],
		"Leaky"->LeakyReLU[0.2],
		"Norm"->NetInitialize@norm,
		"Style"->styleModify[name<>"/Conv0_up",c,s,w2]
	};
	path={
		NetPort["Input"]->"Affine"->"Blur",
		{"Blur",NetPort["Noise"]}->"Noise"->"Leaky"->"Norm",
		{"Norm",NetPort["Style"]}->"Style"->NetPort["Output"]
	};
	"In"->NetGraph[layers,path]
]
getBlockOut[i_,c_,s_,{w1_,w2_}]:=GeneralUtilities`Scope[
	name=StringRiffle[{i,"x",i},""];
	norm=InstanceNormalizationLayer["Epsilon"->10^-8,"Input"->{c,s,s}];
	layers={
		"Affine"->getCN[name<>"/Conv1",w1],
		"Noise"->styleNoise[name<>"/Conv1",c,s],
		"Leaky"->LeakyReLU[0.2],
		"Norm"->NetInitialize@norm,
		"Style"->styleModify[name<>"/Conv1",c,s,w2]
	};
	path={
		NetPort["Input"]->"Affine",
		{"Affine",NetPort["Noise"]}->"Noise"->"Leaky"->"Norm",
		{"Norm",NetPort["Style"]}->"Style"->NetPort["Output"]
	};
	"Out"->NetInitialize@NetGraph[layers,path]
]
getBlock[i_,c_,ws_List]:=GeneralUtilities`Scope[
	layers={
		"ZoomIn"->ResizeLayer[{Scaled[2],Scaled[2]},"Input"->{Automatic,i/2,i/2}],
		"ZoomOut"->ResizeLayer[{i,i}],
		getBlockIn[i,c,i,ws],
		getBlockOut[i,c,i,ws]
	};
	path={
		NetPort["Input"]->"ZoomIn",
		NetPort["Noise"]->"ZoomOut",
		{"ZoomIn","ZoomOut",NetPort["Style"]}->"In",
		{"In","ZoomOut",NetPort["Style"]}->"Out"->NetPort["Output"]
	};
	NetFlatten@NetGraph[layers,path,"Noise"->{1,512,512}]
]


$embedding=NetChain[
	Flatten@{
		PixelNormalizationLayer["Epsilon"->10^-8],
		Table[{embedFC[i],LeakyReLU[0.2]},{i,0,7}]
	},
	"Input"->{512}
]
$embedding[RandomVariate[NormalDistribution[],512]]//Flatten//Histogram


$head=GeneralUtilities`Scope[
	norm=InstanceNormalizationLayer["Epsilon"->10^-8,"Input"->{512,4,4}];
	layers1={
		"Base"->ConstantArrayLayer[
			"Array"->Flatten[Normal@params["G_synthesis/4x4/Const/const"],1]
		],
		"Noise"->styleNoise["4x4/Const",512,4],
		"Leaky"->LeakyReLU[0.2],
		"Norm"->NetInitialize@norm,
		"Style"->styleModify["4x4/Const",512,4,Sqrt[512]]
	};
	path1={
		{"Base",NetPort["Noise"]}->"Noise"->"Leaky"->"Norm",
		{"Norm",NetPort["Style"]}->"Style"->NetPort["Output"]
	};
	layers2={
		"Affine"->getCN["4x4/Conv",48],
		"Noise"->styleNoise["4x4/Conv",512,4],
		"Leaky"->LeakyReLU[0.2],
		"Norm"->NetInitialize@norm,
		"Style"->styleModify["4x4/Conv",512,4,Sqrt@512]
	};
	path2={
		NetPort["Input"]->"Affine",
		{"Affine",NetPort["Noise"]}->"Noise"->"Leaky"->"Norm",
		{"Norm",NetPort["Style"]}->"Style"->NetPort["Output"]
	};
	layers={
		"ZoomOut"->ResizeLayer[{4,4}],
		"In"->NetGraph[layers1,path1],
		"Out"->NetGraph[layers2,path2]
	};
	path={
		NetPort["Noise"]->"ZoomOut",
		{"ZoomOut",NetPort["Style"]}->"In",
		{"In","ZoomOut",NetPort["Style"]}->"Out"->NetPort["Output"]
	};
	NetGraph[layers,path,"Noise"->{1,512,512}]
]


$tail=GeneralUtilities`Scope[
	weight=Normal@params["G_synthesis/ToRGB_lod0/weight"];
	trans=TransposeLayer[{1<->4,2<->3,3<->4}];
	ConvolutionLayer[
		"Weights"->trans@weight/4,
		"Biases"->params["G_synthesis/ToRGB_lod0/bias"]
	]
]


layers=Flatten@{
	Table["Noise_"<>ToString[i]->PartLayer[i;;i],{i,9}],
	"Affine"->$embedding,
	"4x4"->$head,
	"8x8"->getBlock[8,512,{48,Sqrt[512]}],
	"16x16"->getBlock[16,512,{48,Sqrt[512]}],
	"32x32"->getBlock[32,512,{48,Sqrt[512]}],
	"64x64"->getBlock[64,256,{48,Sqrt[512]}],
	"128x128"->getBlock[128,128,{48,Sqrt[512]}],
	"256x256"->getBlock[256,64,{48,Sqrt[512]}],
	"512x512"->getBlock[512,32,{48,Sqrt[512]}],
	"1024x1024"->getBlock[1024,16,{48,Sqrt[512]}],
	"ToRGB"->$tail
};


getName[i_]:=StringRiffle[{i,"x",i},""];
path=Flatten@{
	NetPort["Noise"]->Table["Noise_"<>ToString[i],{i,9}],
	NetPort["Input"]->"Affine",
	{"Noise_1","Affine"}->"4x4",
	Table[{getName[2^i],"Noise_"<>ToString[i],"Affine"}->getName[2^(i+1)],{i,2,9}],
	"1024x1024"->"ToRGB"->NetPort["Output"]
};
mainNet=NetGraph[layers,path]


out=mainNet[<|
	"Input"->RandomVariate[NormalDistribution[],512],
	"Noise"->RandomVariate[NormalDistribution[],{9,512,512}]
|>,
	TargetDevice->"GPU"
];
out//Dimensions
out//Flatten//Histogram
Image[out,Interleaving->False]



