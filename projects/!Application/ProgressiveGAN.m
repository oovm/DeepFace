(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< DeepMath`;


System`SerializeDefinitions = ResourceFunction["BinarySerializeWithDefinitions"];

FunctionRepository`PGGANTrainedOnCelebA`handlerFunction::usage = "";
FunctionRepository`PGGANTrainedOnCelebA`handlerFunction[asc_Association, other___] := Block[
	{
		net = asc["Models", "Main"],
		PGGAN, device, size, map
	},
	Options[PGGAN] = {TargetDevice -> "GPU", ImageSize -> 384};
	PGGAN[] := PGGAN[1];
	PGGAN[n_?Internal`PositiveIntegerQ, opt : OptionsPattern[]] := PGGAN[RandomVariate[NormalDistribution[], {n, 512}], opt];
	PGGAN[n_?VectorQ, opt : OptionsPattern[]] := PGGAN[{n}, opt];
	PGGAN[n_?MatrixQ, opt : OptionsPattern[]] := (
		{device, size} = OptionValue[{TargetDevice, ImageSize}];
		map = <|
			"Input" -> RawArray["Real64", #],
			"Output" -> ImageResize[net[#, TargetDevice -> device], size]
		|>&;
		map /@ n
	);
	PGGAN[___] := Message[DeepMath`NetApplication::illInput];
	PGGAN[other]
];
app = DeepMath`NetApplication@<|
	"Name" -> "PGGAN trained on CelebA",
	"Input" -> "Normal Distribution Vector<512>",
	"Example" -> Inactive[RandomVariate][NormalDistribution[], 512],
	"Date" -> DateString[],
	"Handler" -> FunctionRepository`PGGANTrainedOnCelebA`handlerFunction,
	"Models" -> <|
		"Main" -> Import@"PGGAN trained on CelebA.WXF"
	|>
|>;
byte = System`SerializeDefinitions[app];
Export["PGGAN trained on CelebA.app", Unevaluated[BinaryDeserialize@byte], "WXF"]
