÷С
МЁ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ћњ
~
firstlayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namefirstlayer/kernel
w
%firstlayer/kernel/Read/ReadVariableOpReadVariableOpfirstlayer/kernel*
_output_shapes

:*
dtype0
v
firstlayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefirstlayer/bias
o
#firstlayer/bias/Read/ReadVariableOpReadVariableOpfirstlayer/bias*
_output_shapes
:*
dtype0
А
secondlayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_namesecondlayer/kernel
y
&secondlayer/kernel/Read/ReadVariableOpReadVariableOpsecondlayer/kernel*
_output_shapes

:*
dtype0
x
secondlayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namesecondlayer/bias
q
$secondlayer/bias/Read/ReadVariableOpReadVariableOpsecondlayer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
М
Adam/firstlayer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/firstlayer/kernel/m
Е
,Adam/firstlayer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/firstlayer/kernel/m*
_output_shapes

:*
dtype0
Д
Adam/firstlayer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/firstlayer/bias/m
}
*Adam/firstlayer/bias/m/Read/ReadVariableOpReadVariableOpAdam/firstlayer/bias/m*
_output_shapes
:*
dtype0
О
Adam/secondlayer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/secondlayer/kernel/m
З
-Adam/secondlayer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/secondlayer/kernel/m*
_output_shapes

:*
dtype0
Ж
Adam/secondlayer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/secondlayer/bias/m

+Adam/secondlayer/bias/m/Read/ReadVariableOpReadVariableOpAdam/secondlayer/bias/m*
_output_shapes
:*
dtype0
М
Adam/firstlayer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/firstlayer/kernel/v
Е
,Adam/firstlayer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/firstlayer/kernel/v*
_output_shapes

:*
dtype0
Д
Adam/firstlayer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/firstlayer/bias/v
}
*Adam/firstlayer/bias/v/Read/ReadVariableOpReadVariableOpAdam/firstlayer/bias/v*
_output_shapes
:*
dtype0
О
Adam/secondlayer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/secondlayer/kernel/v
З
-Adam/secondlayer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/secondlayer/kernel/v*
_output_shapes

:*
dtype0
Ж
Adam/secondlayer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/secondlayer/bias/v

+Adam/secondlayer/bias/v/Read/ReadVariableOpReadVariableOpAdam/secondlayer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
с"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ђ"
valueҐ"BЯ" BШ"
Ъ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
¶

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¶

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
М
iter

beta_1

beta_2
	decay
 learning_ratem<m=m>m?v@vAvBvC*
 
0
1
2
3*
 
0
1
2
3*
* 
∞
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

&serving_default* 
a[
VARIABLE_VALUEfirstlayer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEfirstlayer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
У
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUEsecondlayer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEsecondlayer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
У
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

10
21*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	3total
	4count
5	variables
6	keras_api*
H
	7total
	8count
9
_fn_kwargs
:	variables
;	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

5	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

70
81*

:	variables*
Д~
VARIABLE_VALUEAdam/firstlayer/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/firstlayer/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdam/secondlayer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/secondlayer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/firstlayer/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/firstlayer/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdam/secondlayer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/secondlayer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г
 serving_default_firstlayer_inputPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Н
StatefulPartitionedCallStatefulPartitionedCall serving_default_firstlayer_inputfirstlayer/kernelfirstlayer/biassecondlayer/kernelsecondlayer/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_932457
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ѕ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%firstlayer/kernel/Read/ReadVariableOp#firstlayer/bias/Read/ReadVariableOp&secondlayer/kernel/Read/ReadVariableOp$secondlayer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/firstlayer/kernel/m/Read/ReadVariableOp*Adam/firstlayer/bias/m/Read/ReadVariableOp-Adam/secondlayer/kernel/m/Read/ReadVariableOp+Adam/secondlayer/bias/m/Read/ReadVariableOp,Adam/firstlayer/kernel/v/Read/ReadVariableOp*Adam/firstlayer/bias/v/Read/ReadVariableOp-Adam/secondlayer/kernel/v/Read/ReadVariableOp+Adam/secondlayer/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_932583
¶
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefirstlayer/kernelfirstlayer/biassecondlayer/kernelsecondlayer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/firstlayer/kernel/mAdam/firstlayer/bias/mAdam/secondlayer/kernel/mAdam/secondlayer/bias/mAdam/firstlayer/kernel/vAdam/firstlayer/bias/vAdam/secondlayer/kernel/vAdam/secondlayer/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_932656ув
ш
е
F__inference_sequential_layer_call_and_return_conditional_losses_932424

inputs;
)firstlayer_matmul_readvariableop_resource:8
*firstlayer_biasadd_readvariableop_resource:<
*secondlayer_matmul_readvariableop_resource:9
+secondlayer_biasadd_readvariableop_resource:
identityИҐ!firstlayer/BiasAdd/ReadVariableOpҐ firstlayer/MatMul/ReadVariableOpҐ"secondlayer/BiasAdd/ReadVariableOpҐ!secondlayer/MatMul/ReadVariableOpК
 firstlayer/MatMul/ReadVariableOpReadVariableOp)firstlayer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
firstlayer/MatMulMatMulinputs(firstlayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€И
!firstlayer/BiasAdd/ReadVariableOpReadVariableOp*firstlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
firstlayer/BiasAddBiasAddfirstlayer/MatMul:product:0)firstlayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
firstlayer/ReluRelufirstlayer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€М
!secondlayer/MatMul/ReadVariableOpReadVariableOp*secondlayer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ш
secondlayer/MatMulMatMulfirstlayer/Relu:activations:0)secondlayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€К
"secondlayer/BiasAdd/ReadVariableOpReadVariableOp+secondlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ъ
secondlayer/BiasAddBiasAddsecondlayer/MatMul:product:0*secondlayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€n
secondlayer/SoftmaxSoftmaxsecondlayer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
IdentityIdentitysecondlayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€÷
NoOpNoOp"^firstlayer/BiasAdd/ReadVariableOp!^firstlayer/MatMul/ReadVariableOp#^secondlayer/BiasAdd/ReadVariableOp"^secondlayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2F
!firstlayer/BiasAdd/ReadVariableOp!firstlayer/BiasAdd/ReadVariableOp2D
 firstlayer/MatMul/ReadVariableOp firstlayer/MatMul/ReadVariableOp2H
"secondlayer/BiasAdd/ReadVariableOp"secondlayer/BiasAdd/ReadVariableOp2F
!secondlayer/MatMul/ReadVariableOp!secondlayer/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Э

ч
F__inference_firstlayer_layer_call_and_return_conditional_losses_932238

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Є
Ў
+__inference_sequential_layer_call_fn_932346
firstlayer_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallfirstlayer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_932322o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefirstlayer_input
—
Њ
F__inference_sequential_layer_call_and_return_conditional_losses_932262

inputs#
firstlayer_932239:
firstlayer_932241:$
secondlayer_932256: 
secondlayer_932258:
identityИҐ"firstlayer/StatefulPartitionedCallҐ#secondlayer/StatefulPartitionedCallш
"firstlayer/StatefulPartitionedCallStatefulPartitionedCallinputsfirstlayer_932239firstlayer_932241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_firstlayer_layer_call_and_return_conditional_losses_932238°
#secondlayer/StatefulPartitionedCallStatefulPartitionedCall+firstlayer/StatefulPartitionedCall:output:0secondlayer_932256secondlayer_932258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_secondlayer_layer_call_and_return_conditional_losses_932255{
IdentityIdentity,secondlayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp#^firstlayer/StatefulPartitionedCall$^secondlayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2H
"firstlayer/StatefulPartitionedCall"firstlayer/StatefulPartitionedCall2J
#secondlayer/StatefulPartitionedCall#secondlayer/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ
ќ
+__inference_sequential_layer_call_fn_932393

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_932262o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
£

ш
G__inference_secondlayer_layer_call_and_return_conditional_losses_932255

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
п
»
F__inference_sequential_layer_call_and_return_conditional_losses_932374
firstlayer_input#
firstlayer_932363:
firstlayer_932365:$
secondlayer_932368: 
secondlayer_932370:
identityИҐ"firstlayer/StatefulPartitionedCallҐ#secondlayer/StatefulPartitionedCallВ
"firstlayer/StatefulPartitionedCallStatefulPartitionedCallfirstlayer_inputfirstlayer_932363firstlayer_932365*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_firstlayer_layer_call_and_return_conditional_losses_932238°
#secondlayer/StatefulPartitionedCallStatefulPartitionedCall+firstlayer/StatefulPartitionedCall:output:0secondlayer_932368secondlayer_932370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_secondlayer_layer_call_and_return_conditional_losses_932255{
IdentityIdentity,secondlayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp#^firstlayer/StatefulPartitionedCall$^secondlayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2H
"firstlayer/StatefulPartitionedCall"firstlayer/StatefulPartitionedCall2J
#secondlayer/StatefulPartitionedCall#secondlayer/StatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefirstlayer_input
М
—
$__inference_signature_wrapper_932457
firstlayer_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallfirstlayer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_932220o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefirstlayer_input
Є
Ў
+__inference_sequential_layer_call_fn_932273
firstlayer_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallfirstlayer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_932262o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefirstlayer_input
∆U
Є
"__inference__traced_restore_932656
file_prefix4
"assignvariableop_firstlayer_kernel:0
"assignvariableop_1_firstlayer_bias:7
%assignvariableop_2_secondlayer_kernel:1
#assignvariableop_3_secondlayer_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: >
,assignvariableop_13_adam_firstlayer_kernel_m:8
*assignvariableop_14_adam_firstlayer_bias_m:?
-assignvariableop_15_adam_secondlayer_kernel_m:9
+assignvariableop_16_adam_secondlayer_bias_m:>
,assignvariableop_17_adam_firstlayer_kernel_v:8
*assignvariableop_18_adam_firstlayer_bias_v:?
-assignvariableop_19_adam_secondlayer_kernel_v:9
+assignvariableop_20_adam_secondlayer_bias_v:
identity_22ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Њ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*д

valueЏ
B„
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B М
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOpAssignVariableOp"assignvariableop_firstlayer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_1AssignVariableOp"assignvariableop_1_firstlayer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_2AssignVariableOp%assignvariableop_2_secondlayer_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_3AssignVariableOp#assignvariableop_3_secondlayer_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_13AssignVariableOp,assignvariableop_13_adam_firstlayer_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_firstlayer_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_secondlayer_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_secondlayer_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_firstlayer_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_firstlayer_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_19AssignVariableOp-assignvariableop_19_adam_secondlayer_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_secondlayer_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Э
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: К
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Э

ч
F__inference_firstlayer_layer_call_and_return_conditional_losses_932477

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
Щ
,__inference_secondlayer_layer_call_fn_932486

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_secondlayer_layer_call_and_return_conditional_losses_932255o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ
ќ
+__inference_sequential_layer_call_fn_932406

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_932322o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
„
Ґ
!__inference__wrapped_model_932220
firstlayer_inputF
4sequential_firstlayer_matmul_readvariableop_resource:C
5sequential_firstlayer_biasadd_readvariableop_resource:G
5sequential_secondlayer_matmul_readvariableop_resource:D
6sequential_secondlayer_biasadd_readvariableop_resource:
identityИҐ,sequential/firstlayer/BiasAdd/ReadVariableOpҐ+sequential/firstlayer/MatMul/ReadVariableOpҐ-sequential/secondlayer/BiasAdd/ReadVariableOpҐ,sequential/secondlayer/MatMul/ReadVariableOp†
+sequential/firstlayer/MatMul/ReadVariableOpReadVariableOp4sequential_firstlayer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Я
sequential/firstlayer/MatMulMatMulfirstlayer_input3sequential/firstlayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
,sequential/firstlayer/BiasAdd/ReadVariableOpReadVariableOp5sequential_firstlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Є
sequential/firstlayer/BiasAddBiasAdd&sequential/firstlayer/MatMul:product:04sequential/firstlayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
sequential/firstlayer/ReluRelu&sequential/firstlayer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
,sequential/secondlayer/MatMul/ReadVariableOpReadVariableOp5sequential_secondlayer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0є
sequential/secondlayer/MatMulMatMul(sequential/firstlayer/Relu:activations:04sequential/secondlayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
-sequential/secondlayer/BiasAdd/ReadVariableOpReadVariableOp6sequential_secondlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ї
sequential/secondlayer/BiasAddBiasAdd'sequential/secondlayer/MatMul:product:05sequential/secondlayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
sequential/secondlayer/SoftmaxSoftmax'sequential/secondlayer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€w
IdentityIdentity(sequential/secondlayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€В
NoOpNoOp-^sequential/firstlayer/BiasAdd/ReadVariableOp,^sequential/firstlayer/MatMul/ReadVariableOp.^sequential/secondlayer/BiasAdd/ReadVariableOp-^sequential/secondlayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2\
,sequential/firstlayer/BiasAdd/ReadVariableOp,sequential/firstlayer/BiasAdd/ReadVariableOp2Z
+sequential/firstlayer/MatMul/ReadVariableOp+sequential/firstlayer/MatMul/ReadVariableOp2^
-sequential/secondlayer/BiasAdd/ReadVariableOp-sequential/secondlayer/BiasAdd/ReadVariableOp2\
,sequential/secondlayer/MatMul/ReadVariableOp,sequential/secondlayer/MatMul/ReadVariableOp:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefirstlayer_input
ш
е
F__inference_sequential_layer_call_and_return_conditional_losses_932442

inputs;
)firstlayer_matmul_readvariableop_resource:8
*firstlayer_biasadd_readvariableop_resource:<
*secondlayer_matmul_readvariableop_resource:9
+secondlayer_biasadd_readvariableop_resource:
identityИҐ!firstlayer/BiasAdd/ReadVariableOpҐ firstlayer/MatMul/ReadVariableOpҐ"secondlayer/BiasAdd/ReadVariableOpҐ!secondlayer/MatMul/ReadVariableOpК
 firstlayer/MatMul/ReadVariableOpReadVariableOp)firstlayer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
firstlayer/MatMulMatMulinputs(firstlayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€И
!firstlayer/BiasAdd/ReadVariableOpReadVariableOp*firstlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
firstlayer/BiasAddBiasAddfirstlayer/MatMul:product:0)firstlayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
firstlayer/ReluRelufirstlayer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€М
!secondlayer/MatMul/ReadVariableOpReadVariableOp*secondlayer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ш
secondlayer/MatMulMatMulfirstlayer/Relu:activations:0)secondlayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€К
"secondlayer/BiasAdd/ReadVariableOpReadVariableOp+secondlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ъ
secondlayer/BiasAddBiasAddsecondlayer/MatMul:product:0*secondlayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€n
secondlayer/SoftmaxSoftmaxsecondlayer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
IdentityIdentitysecondlayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€÷
NoOpNoOp"^firstlayer/BiasAdd/ReadVariableOp!^firstlayer/MatMul/ReadVariableOp#^secondlayer/BiasAdd/ReadVariableOp"^secondlayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2F
!firstlayer/BiasAdd/ReadVariableOp!firstlayer/BiasAdd/ReadVariableOp2D
 firstlayer/MatMul/ReadVariableOp firstlayer/MatMul/ReadVariableOp2H
"secondlayer/BiasAdd/ReadVariableOp"secondlayer/BiasAdd/ReadVariableOp2F
!secondlayer/MatMul/ReadVariableOp!secondlayer/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
п
»
F__inference_sequential_layer_call_and_return_conditional_losses_932360
firstlayer_input#
firstlayer_932349:
firstlayer_932351:$
secondlayer_932354: 
secondlayer_932356:
identityИҐ"firstlayer/StatefulPartitionedCallҐ#secondlayer/StatefulPartitionedCallВ
"firstlayer/StatefulPartitionedCallStatefulPartitionedCallfirstlayer_inputfirstlayer_932349firstlayer_932351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_firstlayer_layer_call_and_return_conditional_losses_932238°
#secondlayer/StatefulPartitionedCallStatefulPartitionedCall+firstlayer/StatefulPartitionedCall:output:0secondlayer_932354secondlayer_932356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_secondlayer_layer_call_and_return_conditional_losses_932255{
IdentityIdentity,secondlayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp#^firstlayer/StatefulPartitionedCall$^secondlayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2H
"firstlayer/StatefulPartitionedCall"firstlayer/StatefulPartitionedCall2J
#secondlayer/StatefulPartitionedCall#secondlayer/StatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_namefirstlayer_input
∆
Ш
+__inference_firstlayer_layer_call_fn_932466

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_firstlayer_layer_call_and_return_conditional_losses_932238o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
—
Њ
F__inference_sequential_layer_call_and_return_conditional_losses_932322

inputs#
firstlayer_932311:
firstlayer_932313:$
secondlayer_932316: 
secondlayer_932318:
identityИҐ"firstlayer/StatefulPartitionedCallҐ#secondlayer/StatefulPartitionedCallш
"firstlayer/StatefulPartitionedCallStatefulPartitionedCallinputsfirstlayer_932311firstlayer_932313*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_firstlayer_layer_call_and_return_conditional_losses_932238°
#secondlayer/StatefulPartitionedCallStatefulPartitionedCall+firstlayer/StatefulPartitionedCall:output:0secondlayer_932316secondlayer_932318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_secondlayer_layer_call_and_return_conditional_losses_932255{
IdentityIdentity,secondlayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp#^firstlayer/StatefulPartitionedCall$^secondlayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2H
"firstlayer/StatefulPartitionedCall"firstlayer/StatefulPartitionedCall2J
#secondlayer/StatefulPartitionedCall#secondlayer/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м1
т
__inference__traced_save_932583
file_prefix0
,savev2_firstlayer_kernel_read_readvariableop.
*savev2_firstlayer_bias_read_readvariableop1
-savev2_secondlayer_kernel_read_readvariableop/
+savev2_secondlayer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_firstlayer_kernel_m_read_readvariableop5
1savev2_adam_firstlayer_bias_m_read_readvariableop8
4savev2_adam_secondlayer_kernel_m_read_readvariableop6
2savev2_adam_secondlayer_bias_m_read_readvariableop7
3savev2_adam_firstlayer_kernel_v_read_readvariableop5
1savev2_adam_firstlayer_bias_v_read_readvariableop8
4savev2_adam_secondlayer_kernel_v_read_readvariableop6
2savev2_adam_secondlayer_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ї
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*д

valueЏ
B„
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЩ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B ч
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_firstlayer_kernel_read_readvariableop*savev2_firstlayer_bias_read_readvariableop-savev2_secondlayer_kernel_read_readvariableop+savev2_secondlayer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_firstlayer_kernel_m_read_readvariableop1savev2_adam_firstlayer_bias_m_read_readvariableop4savev2_adam_secondlayer_kernel_m_read_readvariableop2savev2_adam_secondlayer_bias_m_read_readvariableop3savev2_adam_firstlayer_kernel_v_read_readvariableop1savev2_adam_firstlayer_bias_v_read_readvariableop4savev2_adam_secondlayer_kernel_v_read_readvariableop2savev2_adam_secondlayer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Й
_input_shapesx
v: ::::: : : : : : : : : ::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
£

ш
G__inference_secondlayer_layer_call_and_return_conditional_losses_932497

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"џL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ј
serving_defaultђ
M
firstlayer_input9
"serving_default_firstlayer_input:0€€€€€€€€€?
secondlayer0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ВA
і
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_sequential
ї

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ы
iter

beta_1

beta_2
	decay
 learning_ratem<m=m>m?v@vAvBvC"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
ъ2ч
+__inference_sequential_layer_call_fn_932273
+__inference_sequential_layer_call_fn_932393
+__inference_sequential_layer_call_fn_932406
+__inference_sequential_layer_call_fn_932346ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
F__inference_sequential_layer_call_and_return_conditional_losses_932424
F__inference_sequential_layer_call_and_return_conditional_losses_932442
F__inference_sequential_layer_call_and_return_conditional_losses_932360
F__inference_sequential_layer_call_and_return_conditional_losses_932374ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
’B“
!__inference__wrapped_model_932220firstlayer_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
,
&serving_default"
signature_map
#:!2firstlayer/kernel
:2firstlayer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
’2“
+__inference_firstlayer_layer_call_fn_932466Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_firstlayer_layer_call_and_return_conditional_losses_932477Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
$:"2secondlayer/kernel
:2secondlayer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
÷2”
,__inference_secondlayer_layer_call_fn_932486Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_secondlayer_layer_call_and_return_conditional_losses_932497Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
‘B—
$__inference_signature_wrapper_932457firstlayer_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	3total
	4count
5	variables
6	keras_api"
_tf_keras_metric
^
	7total
	8count
9
_fn_kwargs
:	variables
;	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
30
41"
trackable_list_wrapper
-
5	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
-
:	variables"
_generic_user_object
(:&2Adam/firstlayer/kernel/m
": 2Adam/firstlayer/bias/m
):'2Adam/secondlayer/kernel/m
#:!2Adam/secondlayer/bias/m
(:&2Adam/firstlayer/kernel/v
": 2Adam/firstlayer/bias/v
):'2Adam/secondlayer/kernel/v
#:!2Adam/secondlayer/bias/v°
!__inference__wrapped_model_932220|9Ґ6
/Ґ,
*К'
firstlayer_input€€€€€€€€€
™ "9™6
4
secondlayer%К"
secondlayer€€€€€€€€€¶
F__inference_firstlayer_layer_call_and_return_conditional_losses_932477\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
+__inference_firstlayer_layer_call_fn_932466O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€І
G__inference_secondlayer_layer_call_and_return_conditional_losses_932497\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ 
,__inference_secondlayer_layer_call_fn_932486O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Ї
F__inference_sequential_layer_call_and_return_conditional_losses_932360pAҐ>
7Ґ4
*К'
firstlayer_input€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
F__inference_sequential_layer_call_and_return_conditional_losses_932374pAҐ>
7Ґ4
*К'
firstlayer_input€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ∞
F__inference_sequential_layer_call_and_return_conditional_losses_932424f7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ∞
F__inference_sequential_layer_call_and_return_conditional_losses_932442f7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Т
+__inference_sequential_layer_call_fn_932273cAҐ>
7Ґ4
*К'
firstlayer_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€Т
+__inference_sequential_layer_call_fn_932346cAҐ>
7Ґ4
*К'
firstlayer_input€€€€€€€€€
p

 
™ "К€€€€€€€€€И
+__inference_sequential_layer_call_fn_932393Y7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€И
+__inference_sequential_layer_call_fn_932406Y7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€є
$__inference_signature_wrapper_932457РMҐJ
Ґ 
C™@
>
firstlayer_input*К'
firstlayer_input€€€€€€€€€"9™6
4
secondlayer%К"
secondlayer€€€€€€€€€