#!/usr/bin/env python3
"""Recompile one PyTorch translation unit and relink an isolated libtorch_cpu."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('/root/benchmark/pytorch-kunpeng'))
    args=p.parse_args();source=args.project/'pytorch-v2.5.0';build=source/'build'
    out=ROOT/'bin/kml_schedules';out.mkdir(parents=True,exist_ok=True)
    original=source/'aten/src/ATen/native/LinearAlgebra.cpp'
    patch=ROOT/'kml_bmm_schedule.patch';target=out/'libtorch_cpu.so'
    inputs={'original_source':sha(original),'patch':sha(patch),'original_library':sha(source/'torch/lib/libtorch_cpu.so')}
    if (out/'build.json').exists():
        previous=json.loads((out/'build.json').read_text())
        if previous['inputs']==inputs and target.exists() and sha(target)==previous['library_sha256']:
            print(target);return
        raise RuntimeError('existing isolated build differs; preserve it and choose a new build directory')
    derived=out/'LinearAlgebra.cpp'
    with (out/'patch.log').open('w') as log:
        subprocess.run(['patch','--batch','--output',str(derived),str(original),str(patch)],stdout=log,stderr=subprocess.STDOUT,check=True)
    object_target='caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/LinearAlgebra.cpp.o'
    def command(name):
        result=subprocess.run(['ninja','-t','commands',name],cwd=build,capture_output=True,text=True,check=True)
        return shlex.split(result.stdout.splitlines()[-1])
    compile_cmd=command(object_target)
    for flag,path in [('-o',out/'LinearAlgebra.cpp.o'),('-MF',out/'LinearAlgebra.cpp.o.d'),('-MT',out/'LinearAlgebra.cpp.o')]:
        compile_cmd[compile_cmd.index(flag)+1]=str(path)
    compile_cmd[compile_cmd.index(str(original))]=str(derived)
    link_cmd=command('lib/libtorch_cpu.so')
    assert link_cmd[:2]==[':','&&']
    link_cmd=link_cmd[2:link_cmd.index('&&',2)]
    link_cmd[link_cmd.index('-o')+1]=str(target)
    link_cmd[link_cmd.index(object_target)]=str(out/'LinearAlgebra.cpp.o')
    # Keep the original object files, compiler and link flags; add the dependency path
    # needed by the isolated location. The selected KML multi is loaded before it.
    link_cmd.append('-Wl,-rpath,'+str(source/'torch/lib'))
    env=os.environ.copy();compiler=args.project/'hpckit-components/BiShengCompiler-5.2.0-aarch64-linux-mini-hpc'
    env['LD_LIBRARY_PATH']=':'.join([str(compiler/'lib'),str(compiler/'lib/aarch64-unknown-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    for name,cmd in [('compile',compile_cmd),('link',link_cmd)]:
        print(name+' isolated schedule library',flush=True)
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert sha(original)==inputs['original_source']
    assert sha(source/'torch/lib/libtorch_cpu.so')==inputs['original_library']
    (out/'build.json').write_text(json.dumps({'inputs':inputs,'compile_command':compile_cmd,'link_command':link_cmd,
        'library_sha256':sha(target),'derived_source_sha256':sha(derived),'scope':'only BMM outer-vs-inner branch changed; installed torch untouched'},indent=2)+'\n')
    print(target,flush=True)

if __name__=='__main__':main()
