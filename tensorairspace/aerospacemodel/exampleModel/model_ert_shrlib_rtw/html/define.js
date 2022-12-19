function CodeDefine() { 
this.def = new Array();
this.def["rt_OneStep"] = {file: "ert_main_c.html",line:33,type:"fcn"};
this.def["main"] = {file: "ert_main_c.html",line:70,type:"fcn"};
this.def["model_B"] = {file: "model_c.html",line:20,type:"var"};
this.def["model_X"] = {file: "model_c.html",line:23,type:"var"};
this.def["model_U"] = {file: "model_c.html",line:26,type:"var"};
this.def["model_Y"] = {file: "model_c.html",line:29,type:"var"};
this.def["model_M_"] = {file: "model_c.html",line:32,type:"var"};
this.def["model_M"] = {file: "model_c.html",line:33,type:"var"};
this.def["rt_ertODEUpdateContinuousStates"] = {file: "model_c.html",line:39,type:"fcn"};
this.def["model_step"] = {file: "model_c.html",line:117,type:"fcn"};
this.def["model_derivatives"] = {file: "model_c.html",line:266,type:"fcn"};
this.def["model_initialize"] = {file: "model_c.html",line:314,type:"fcn"};
this.def["model_terminate"] = {file: "model_c.html",line:395,type:"fcn"};
this.def["B_model_T"] = {file: "model_h.html",line:63,type:"type"};
this.def["X_model_T"] = {file: "model_h.html",line:75,type:"type"};
this.def["XDot_model_T"] = {file: "model_h.html",line:87,type:"type"};
this.def["XDis_model_T"] = {file: "model_h.html",line:99,type:"type"};
this.def["ODE3_IntgData"] = {file: "model_h.html",line:108,type:"type"};
this.def["ExtU_model_T"] = {file: "model_h.html",line:115,type:"type"};
this.def["ExtY_model_T"] = {file: "model_h.html",line:124,type:"type"};
this.def["RT_MODEL_model_T"] = {file: "model_types_h.html",line:20,type:"type"};
this.def["int8_T"] = {file: "rtwtypes_h.html",line:47,type:"type"};
this.def["uint8_T"] = {file: "rtwtypes_h.html",line:48,type:"type"};
this.def["int16_T"] = {file: "rtwtypes_h.html",line:49,type:"type"};
this.def["uint16_T"] = {file: "rtwtypes_h.html",line:50,type:"type"};
this.def["int32_T"] = {file: "rtwtypes_h.html",line:51,type:"type"};
this.def["uint32_T"] = {file: "rtwtypes_h.html",line:52,type:"type"};
this.def["int64_T"] = {file: "rtwtypes_h.html",line:53,type:"type"};
this.def["uint64_T"] = {file: "rtwtypes_h.html",line:54,type:"type"};
this.def["real32_T"] = {file: "rtwtypes_h.html",line:55,type:"type"};
this.def["real64_T"] = {file: "rtwtypes_h.html",line:56,type:"type"};
this.def["real_T"] = {file: "rtwtypes_h.html",line:62,type:"type"};
this.def["time_T"] = {file: "rtwtypes_h.html",line:63,type:"type"};
this.def["boolean_T"] = {file: "rtwtypes_h.html",line:64,type:"type"};
this.def["int_T"] = {file: "rtwtypes_h.html",line:65,type:"type"};
this.def["uint_T"] = {file: "rtwtypes_h.html",line:66,type:"type"};
this.def["ulong_T"] = {file: "rtwtypes_h.html",line:67,type:"type"};
this.def["ulonglong_T"] = {file: "rtwtypes_h.html",line:68,type:"type"};
this.def["char_T"] = {file: "rtwtypes_h.html",line:69,type:"type"};
this.def["uchar_T"] = {file: "rtwtypes_h.html",line:70,type:"type"};
this.def["byte_T"] = {file: "rtwtypes_h.html",line:71,type:"type"};
this.def["creal32_T"] = {file: "rtwtypes_h.html",line:81,type:"type"};
this.def["creal64_T"] = {file: "rtwtypes_h.html",line:86,type:"type"};
this.def["creal_T"] = {file: "rtwtypes_h.html",line:91,type:"type"};
this.def["cint8_T"] = {file: "rtwtypes_h.html",line:98,type:"type"};
this.def["cuint8_T"] = {file: "rtwtypes_h.html",line:105,type:"type"};
this.def["cint16_T"] = {file: "rtwtypes_h.html",line:112,type:"type"};
this.def["cuint16_T"] = {file: "rtwtypes_h.html",line:119,type:"type"};
this.def["cint32_T"] = {file: "rtwtypes_h.html",line:126,type:"type"};
this.def["cuint32_T"] = {file: "rtwtypes_h.html",line:133,type:"type"};
this.def["cint64_T"] = {file: "rtwtypes_h.html",line:140,type:"type"};
this.def["cuint64_T"] = {file: "rtwtypes_h.html",line:147,type:"type"};
this.def["pointer_T"] = {file: "rtwtypes_h.html",line:168,type:"type"};
}
CodeDefine.instance = new CodeDefine();
var testHarnessInfo = {OwnerFileName: "", HarnessOwner: "", HarnessName: "", IsTestHarness: "0"};
var relPathToBuildDir = "../ert_main.c";
var fileSep = "\\";
var isPC = true;
function Html2SrcLink() {
	this.html2SrcPath = new Array;
	this.html2Root = new Array;
	this.html2SrcPath["ert_main_c.html"] = "../ert_main.c";
	this.html2Root["ert_main_c.html"] = "ert_main_c.html";
	this.html2SrcPath["model_c.html"] = "../model.c";
	this.html2Root["model_c.html"] = "model_c.html";
	this.html2SrcPath["model_h.html"] = "../model.h";
	this.html2Root["model_h.html"] = "model_h.html";
	this.html2SrcPath["model_private_h.html"] = "../model_private.h";
	this.html2Root["model_private_h.html"] = "model_private_h.html";
	this.html2SrcPath["model_types_h.html"] = "../model_types.h";
	this.html2Root["model_types_h.html"] = "model_types_h.html";
	this.html2SrcPath["rtwtypes_h.html"] = "../rtwtypes.h";
	this.html2Root["rtwtypes_h.html"] = "rtwtypes_h.html";
	this.getLink2Src = function (htmlFileName) {
		 if (this.html2SrcPath[htmlFileName])
			 return this.html2SrcPath[htmlFileName];
		 else
			 return null;
	}
	this.getLinkFromRoot = function (htmlFileName) {
		 if (this.html2Root[htmlFileName])
			 return this.html2Root[htmlFileName];
		 else
			 return null;
	}
}
Html2SrcLink.instance = new Html2SrcLink();
var fileList = [
"ert_main_c.html","model_c.html","model_h.html","model_private_h.html","model_types_h.html","rtwtypes_h.html"];
