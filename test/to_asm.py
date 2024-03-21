import subprocess
import os
import re

digit_pattern = r'\b0x[a-fA-F0-9]+\b'# binary codes in Hexadecimal
zeros_pattern = r'^0+\s'#0s
OPT = ["O0", "O1", "O2", "O3"]
fileName = 'test/EnablePacketEncryption.bin'
with open(fileName+'.c','r') as f:#original file
    c_func = f.read()
for opt_state in OPT:
    output_file = fileName +'_' + opt_state
    input_file = fileName+'.c'
    compile_command = f'gcc -c -o {output_file}.o {input_file} -{opt_state} -lm'#compile the code with GCC on Linux
    subprocess.run(compile_command, shell=True, check=True)
    compile_command = f'objdump -d {output_file}.o > {output_file}.s'#disassemble the binary file into assembly instructions
    subprocess.run(compile_command, shell=True, check=True)
    
    input_asm = ''
    with open(output_file+'.s') as f:#asm file
        asm= f.read()
    asm = asm.split('Disassembly of section .text:')[-1].strip()
    for tmp in asm.split('\n'):
        tmp_asm = tmp.split('\t')[-1]#remove the binary code
        tmp_asm = tmp_asm.split('#')[0].strip()#remove the comments
        input_asm+=tmp_asm+'\n'
    input_asm = re.sub(zeros_pattern, '', input_asm)
    before = f"# This is the assembly code with {opt_state} optimization:\n"#prompt
    after = "\n# What is the source code?\n"#prompt
    input_asm_prompt = before+input_asm.strip()+after
    with open(fileName +'_' + opt_state +'.asm','w',encoding='utf-8') as f:
        f.write(input_asm_prompt)

