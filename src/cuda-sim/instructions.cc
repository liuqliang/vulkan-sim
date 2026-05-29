// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Jimmy Kwa, George L. Yuan
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include "instructions.h"
#include "half.h"
#include "half.hpp"
#include "opcodes.h"
#include "ptx_ir.h"
#include "ptx_sim.h"
typedef void *yyscan_t;
class ptx_recognizer;
#include <assert.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/shader.h"
#include "cuda-math.h"
#include "cuda_device_printf.h"
#include "ptx.tab.h"
#include "ptx_loader.h"
#include "vulkan_ray_tracing.h"
#include "vulkan_rt_thread_data.h"

// Jin: include device runtime for CDP
#include "cuda_device_runtime.h"

#include <stdarg.h>
#include "../../libcuda/gpgpu_context.h"

using half_float::half;

const char *g_opcode_string[NUM_OPCODES] = {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) STR,
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) STR,
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
};
// Using profiled information::check the TensorCoreMatrixArrangement.xls for
// details
unsigned thread_group_offset(int thread, unsigned wmma_type,
                             unsigned wmma_layout, unsigned type, int stride) {
  unsigned offset;
  unsigned load_a_row[8] = {0, 128, 0, 128, 64, 192, 64, 192};
  unsigned load_a_col[8] = {0, 8, 0, 8, 4, 12, 4, 12};
  unsigned load_b_row[8] = {0, 8, 0, 8, 4, 12, 4, 12};
  unsigned load_b_col[8] = {0, 128, 0, 128, 64, 192, 64, 192};
  unsigned load_c_float_row[8] = {0, 128, 8, 136, 64, 192, 72, 200};
  unsigned load_c_float_col[8] = {0, 8, 128, 136, 4, 12, 132, 140};
  unsigned load_c_half_row[8] = {0, 128, 8, 136, 64, 192, 72, 200};
  unsigned load_c_half_col[8] = {0, 8, 128, 136, 4, 12, 132, 140};
  unsigned thread_group = thread / 4;
  unsigned in_tg_index = thread % 4;

  switch (wmma_type) {
    case LOAD_A:
      if (wmma_layout == ROW)
        offset = load_a_row[thread_group] + 16 * in_tg_index;
      else
        offset = load_a_col[thread_group] + 16 * in_tg_index;
      break;

    case LOAD_B:
      if (wmma_layout == ROW)
        offset = load_b_row[thread_group] + 16 * in_tg_index;
      else
        offset = load_b_col[thread_group] + 16 * in_tg_index;
      break;

    case LOAD_C:
    case STORE_D:
      if (type == F16_TYPE) {
        if (wmma_layout == ROW)
          offset = load_c_half_row[thread_group] + 16 * in_tg_index;
        else
          offset = load_c_half_col[thread_group] + in_tg_index;
      } else {
        if (wmma_layout == ROW)
          offset = load_c_float_row[thread_group];
        else
          offset = load_c_float_col[thread_group];

        switch (in_tg_index) {
          case 0:
            break;
          case 1:
            if (wmma_layout == ROW)
              offset += 16;
            else
              offset += 1;
            break;
          case 2:
            if (wmma_layout == ROW)
              offset += 2;
            else
              offset += 32;
            break;
          case 3:
            if (wmma_layout == ROW)
              offset += 18;
            else
              offset += 33;
            break;
          default:
            abort();
        }
      }
      break;

    default:
      abort();
  }
  offset = (offset / 16) * stride + offset % 16;
  return offset;
}

int acc_float_offset(int index, int wmma_layout, int stride) {
  int c_row_offset[] = {0, 1, 32, 33, 4, 5, 36, 37};
  int c_col_offset[] = {0, 16, 2, 18, 64, 80, 66, 82};
  int offset;

  if (wmma_layout == ROW)
    offset = c_row_offset[index];
  else if (wmma_layout == COL)
    offset = c_col_offset[index];
  else {
    printf("wrong layout");
    abort();
  }
  offset = (offset / 16) * stride + offset % 16;
  return offset;
}

void inst_not_implemented(const ptx_instruction *pI);
ptx_reg_t srcOperandModifiers(ptx_reg_t opData, operand_info opInfo,
                              operand_info dstInfo, unsigned type,
                              ptx_thread_info *thread);
                              
void video_mem_instruction(const ptx_instruction *pI, ptx_thread_info *thread, int op_code);

void sign_extend(ptx_reg_t &data, unsigned src_size, const operand_info &dst);

void ptx_thread_info::set_reg(const symbol *reg, const ptx_reg_t &value) {
  assert(reg != NULL);
  if (reg->name() == "_") return;
  assert(!m_regs.empty());
  assert(reg->uid() > 0);
  m_regs.back()[reg] = value;
  if (m_enable_debug_trace) m_debug_trace_regs_modified.back()[reg] = value;
  m_last_set_operand_value = value;
}

void ptx_thread_info::print_reg_thread(char *fname) {
  FILE *fp = fopen(fname, "w");
  assert(fp != NULL);

  int size = m_regs.size();

  if (size > 0) {
    reg_map_t reg = m_regs.back();

    reg_map_t::const_iterator it;
    for (it = reg.begin(); it != reg.end(); ++it) {
      const std::string &name = it->first->name();
      const std::string &dec = it->first->decl_location();
      unsigned size = it->first->get_size_in_bytes();
      fprintf(fp, "%s %llu %s %d\n", name.c_str(), it->second, dec.c_str(),
              size);
    }
    // m_regs.pop_back();
  }
  fclose(fp);
}

void ptx_thread_info::resume_reg_thread(char *fname, symbol_table *symtab) {
  FILE *fp2 = fopen(fname, "r");
  assert(fp2 != NULL);
  // m_regs.push_back( reg_map_t() );
  char line[200];
  while (fgets(line, sizeof line, fp2) != NULL) {
    symbol *reg;
    char *pch;
    pch = strtok(line, " ");
    char *name = pch;
    reg = symtab->lookup(name);
    ptx_reg_t data;
    pch = strtok(NULL, " ");
    data = atoi(pch);
    pch = strtok(NULL, " ");
    pch = strtok(NULL, " ");
    m_regs.back()[reg] = data;
  }
  fclose(fp2);
}

ptx_reg_t ptx_thread_info::get_reg(const symbol *reg) {
  static bool unfound_register_warned = false;
  assert(reg != NULL);
  assert(!m_regs.empty());
  reg_map_t::iterator regs_iter = m_regs.back().find(reg);
  if (regs_iter == m_regs.back().end()) {
    assert(reg->type()->get_key().is_reg());
    const std::string &name = reg->name();
    unsigned call_uid = m_callstack.back().m_call_uid;
    ptx_reg_t uninit_reg;
    uninit_reg.u32 = 0x0;
    set_reg(reg, uninit_reg);  // give it a value since we are going to warn the
                               // user anyway
    std::string file_loc = get_location();
    if (!unfound_register_warned) {
      printf(
          "GPGPU-Sim PTX: WARNING (%s) ** reading undefined register \'%s\' "
          "(cuid:%u). Setting to 0X00000000. This is okay if you are "
          "simulating the native ISA"
          "\n",
          file_loc.c_str(), name.c_str(), call_uid);
      unfound_register_warned = true;
    }
    regs_iter = m_regs.back().find(reg);
  }
  if (m_enable_debug_trace)
    m_debug_trace_regs_read.back()[reg] = regs_iter->second;
  return regs_iter->second;
}

ptx_reg_t ptx_thread_info::get_operand_value(const operand_info &op,
                                             operand_info dstInfo,
                                             unsigned opType,
                                             ptx_thread_info *thread,
                                             int derefFlag) {
  ptx_reg_t result, tmp;

  if (op.get_double_operand_type() == 0) {
    if (((opType != BB128_TYPE) && (opType != BB64_TYPE) &&
         (opType != FF64_TYPE)) ||
        (op.get_addr_space() != undefined_space)) {
      if (op.is_reg()) {
        result = get_reg(op.get_symbol());
      } else if (op.is_builtin()) {
        result.u32 = get_builtin(op.get_int(), op.get_addr_offset());
      } else if (op.is_immediate_address()) {
        result.u64 = op.get_addr_offset();
      } else if (op.is_memory_operand()) {
        // a few options here...
        const symbol *sym = op.get_symbol();
        const type_info *type = sym->type();
        const type_info_key &info = type->get_key();

        if (info.is_reg()) {
          const symbol *name = op.get_symbol();
          result.u64 = get_reg(name).u64 + op.get_addr_offset();
        } else if (info.is_param_kernel()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_param_local()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_global()) {
          assert(op.get_addr_offset() == 0);
          result.u64 = sym->get_address();
        } else if (info.is_local()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_const()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (op.is_shared()) {
          result.u64 = op.get_symbol()->get_address() + op.get_addr_offset();
        } else if (op.is_sstarr()) {
          result.u64 = op.get_symbol()->get_address() + op.get_addr_offset();
        } else {
          const char *name = op.name().c_str();
          printf(
              "GPGPU-Sim PTX: ERROR ** get_operand_value : unknown memory "
              "operand type for %s\n",
              name);
          abort();
        }

      } else if (op.is_literal()) {
        result = op.get_literal_value();
      } else if (op.is_label()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_shared()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_sstarr()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_const()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_global()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_local()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_function_address()) {
        result.u64 = (size_t)op.get_symbol()->get_pc();
      } else if (op.is_param_kernel()) {
        result.u64 = op.get_symbol()->get_address();
      } else {
        const char *name = op.name().c_str();
        const symbol *sym2 = op.get_symbol();
        const type_info *type2 = sym2->type();
        const type_info_key &info2 = type2->get_key();
        if (info2.is_param_kernel()) {
          result.u64 = sym2->get_address() + op.get_addr_offset();
        } else {
          printf(
              "GPGPU-Sim PTX: ERROR ** get_operand_value : unknown operand "
              "type for %s\n",
              name);
          assert(0);
        }
      }

      if (op.get_operand_lohi() == 1)
        result.u64 = result.u64 & 0xFFFF;
      else if (op.get_operand_lohi() == 2)
        result.u64 = (result.u64 >> 16) & 0xFFFF;
    } else if (opType == BB128_TYPE) {
      // b128
      result.u128.lowest = get_reg(op.vec_symbol(0)).u32;
      result.u128.low = get_reg(op.vec_symbol(1)).u32;
      result.u128.high = get_reg(op.vec_symbol(2)).u32;
      result.u128.highest = get_reg(op.vec_symbol(3)).u32;
    } else {
      // bb64 or ff64
      result.bits.ls = get_reg(op.vec_symbol(0)).u32;
      result.bits.ms = get_reg(op.vec_symbol(1)).u32;
    }
  } else if (op.get_double_operand_type() == 1) {
    ptx_reg_t firstHalf, secondHalf;
    firstHalf.u64 = get_reg(op.vec_symbol(0)).u64;
    secondHalf.u64 = get_reg(op.vec_symbol(1)).u64;
    if (op.get_operand_lohi() == 1)
      secondHalf.u64 = secondHalf.u64 & 0xFFFF;
    else if (op.get_operand_lohi() == 2)
      secondHalf.u64 = (secondHalf.u64 >> 16) & 0xFFFF;
    result.u64 = firstHalf.u64 + secondHalf.u64;
  } else if (op.get_double_operand_type() == 2) {
    // s[reg1 += reg2]
    // reg1 is incremented after value is returned: the value returned is
    // s[reg1]
    ptx_reg_t firstHalf, secondHalf;
    firstHalf.u64 = get_reg(op.vec_symbol(0)).u64;
    secondHalf.u64 = get_reg(op.vec_symbol(1)).u64;
    if (op.get_operand_lohi() == 1)
      secondHalf.u64 = secondHalf.u64 & 0xFFFF;
    else if (op.get_operand_lohi() == 2)
      secondHalf.u64 = (secondHalf.u64 >> 16) & 0xFFFF;
    result.u64 = firstHalf.u64;
    firstHalf.u64 = firstHalf.u64 + secondHalf.u64;
    set_reg(op.vec_symbol(0), firstHalf);
  } else if (op.get_double_operand_type() == 3) {
    // s[reg += immediate]
    // reg is incremented after value is returned: the value returned is s[reg]
    ptx_reg_t firstHalf;
    firstHalf.u64 = get_reg(op.get_symbol()).u64;
    result.u64 = firstHalf.u64;
    firstHalf.u64 = firstHalf.u64 + op.get_addr_offset();
    set_reg(op.get_symbol(), firstHalf);
  }

  ptx_reg_t finalResult;
  memory_space *mem = NULL;
  size_t size = 0;
  int t = 0;
  finalResult.u64 = 0;

  // complete other cases for reading from memory, such as reading from other
  // const memory
  if ((op.get_addr_space() == global_space) && (derefFlag)) {
    // global memory - g[4], g[$r0]
    mem = thread->get_global_memory();
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = global_space;

    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == shared_space) && (derefFlag)) {
    // shared memory - s[4], s[$r0]
    mem = thread->m_shared_mem;
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = shared_space;

    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == const_space) && (derefFlag)) {
    // const memory - ce0c1[4], ce0c1[$r0]
    mem = thread->get_global_memory();
    type_info_key::type_decode(opType, size, t);
    mem->read((result.u32 + op.get_const_mem_offset()), size / 8,
              &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = const_space;
    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == local_space) && (derefFlag)) {
    // local memory - l0[4], l0[$r0]
    mem = thread->m_local_mem;
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = local_space;
    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else {
    finalResult = result;
  }

  if ((op.get_operand_neg() == true) && (derefFlag)) {
    switch (opType) {
      // Default to f32 for now, need to add support for others
      case S8_TYPE:
      case U8_TYPE:
      case B8_TYPE:
        finalResult.s8 = -finalResult.s8;
        break;
      case S16_TYPE:
      case U16_TYPE:
      case B16_TYPE:
        finalResult.s16 = -finalResult.s16;
        break;
      case S32_TYPE:
      case U32_TYPE:
      case B32_TYPE:
        finalResult.s32 = -finalResult.s32;
        break;
      case S64_TYPE:
      case U64_TYPE:
      case B64_TYPE:
        finalResult.s64 = -finalResult.s64;
        break;
      case F16_TYPE:
        finalResult.f16 = -finalResult.f16;
        break;
      case F32_TYPE:
        finalResult.f32 = -finalResult.f32;
        break;
      case F64_TYPE:
      case FF64_TYPE:
        finalResult.f64 = -finalResult.f64;
        break;
      default:
        assert(0);
    }
  }

  return finalResult;
}

unsigned get_operand_nbits(const operand_info &op) {
  if (op.is_reg()) {
    const symbol *sym = op.get_symbol();
    const type_info *typ = sym->type();
    type_info_key t = typ->get_key();
    switch (t.scalar_type()) {
      case PRED_TYPE:
        return 1;
      case B8_TYPE:
      case S8_TYPE:
      case U8_TYPE:
        return 8;
      case S16_TYPE:
      case U16_TYPE:
      case F16_TYPE:
      case B16_TYPE:
        return 16;
      case S32_TYPE:
      case U32_TYPE:
      case F32_TYPE:
      case B32_TYPE:
        return 32;
      case S64_TYPE:
      case U64_TYPE:
      case F64_TYPE:
      case B64_TYPE:
        return 64;
      default:
        printf("ERROR: unknown register type\n");
        fflush(stdout);
        abort();
    }
  } else {
    printf(
        "ERROR: Need to implement get_operand_nbits() for currently "
        "unsupported operand_info type\n");
    fflush(stdout);
    abort();
  }

  return 0;
}

void ptx_thread_info::get_vector_operand_values(const operand_info &op,
                                                ptx_reg_t *ptx_regs,
                                                unsigned num_elements) {
  assert(op.is_vector());
  assert(num_elements <= 8);

  for (int idx = num_elements - 1; idx >= 0; --idx) {
    const symbol *sym = NULL;
    sym = op.vec_symbol(idx);
    if (strcmp(sym->name().c_str(), "_") != 0) {
      reg_map_t::iterator reg_iter = m_regs.back().find(sym);
      assert(reg_iter != m_regs.back().end());
      ptx_regs[idx] = reg_iter->second;
    }
  }
}

void sign_extend(ptx_reg_t &data, unsigned src_size, const operand_info &dst) {
  if (!dst.is_reg()) return;
  unsigned dst_size = get_operand_nbits(dst);
  if (src_size >= dst_size) return;
  // src_size < dst_size
  unsigned long long mask = 1;
  mask <<= (src_size - 1);
  if ((mask & data.u64) == 0) {
    // no need to sign extend
    return;
  }
  // need to sign extend
  mask = 1;
  mask <<= dst_size - src_size;
  mask -= 1;
  mask <<= src_size;
  data.u64 |= mask;
}

void ptx_thread_info::set_operand_value(const operand_info &dst,
                                        const ptx_reg_t &data, unsigned type,
                                        ptx_thread_info *thread,
                                        const ptx_instruction *pI, int overflow,
                                        int carry) {
  thread->set_operand_value(dst, data, type, thread, pI);

  if (dst.get_double_operand_type() == -2) {
    ptx_reg_t predValue;

    const symbol *sym = dst.vec_symbol(0);
    predValue.u64 = (m_regs.back()[sym].u64) & ~(0x0C);
    predValue.u64 |= ((overflow & 0x01) << 3);
    predValue.u64 |= ((carry & 0x01) << 2);

    set_reg(sym, predValue);
  } else if (dst.get_double_operand_type() == 0) {
    // intentionally do nothing
  } else {
    printf("Unexpected double destination\n");
    assert(0);
  }
}

void ptx_thread_info::set_operand_value(const operand_info &dst,
                                        const ptx_reg_t &data, unsigned type,
                                        ptx_thread_info *thread,
                                        const ptx_instruction *pI) {
  ptx_reg_t dstData;
  memory_space *mem = NULL;
  size_t size;
  int t;

  type_info_key::type_decode(type, size, t);

  /*complete this section for other cases*/
  if (dst.get_addr_space() == undefined_space) {
    ptx_reg_t setValue;
    setValue.u64 = data.u64;

    // Double destination in set instruction ($p0|$p1) - second is negation of
    // first
    if (dst.get_double_operand_type() == -1) {
      ptx_reg_t setValue2;
      const symbol *name1 = dst.vec_symbol(0);
      const symbol *name2 = dst.vec_symbol(1);

      if ((type == F16_TYPE) || (type == F32_TYPE) || (type == F64_TYPE) ||
          (type == FF64_TYPE)) {
        setValue2.f32 = (setValue.u64 == 0) ? 1.0f : 0.0f;
      } else {
        setValue2.u32 = (setValue.u64 == 0) ? 0xFFFFFFFF : 0;
      }

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
    }

    // Double destination in cvt,shr,mul,etc. instruction ($p0|$r4) - second
    // register operand receives data, first predicate operand is set as
    // $p0=($r4!=0) Also for Double destination in set instruction ($p0/$r1)
    else if ((dst.get_double_operand_type() == -2) ||
             (dst.get_double_operand_type() == -3)) {
      ptx_reg_t predValue;
      const symbol *predName = dst.vec_symbol(0);
      const symbol *regName = dst.vec_symbol(1);
      predValue.u64 = 0;

      switch (type) {
        case S8_TYPE:
          if ((setValue.s8 & 0x7F) == 0) predValue.u64 |= 1;
          break;
        case S16_TYPE:
          if ((setValue.s16 & 0x7FFF) == 0) predValue.u64 |= 1;
          break;
        case S32_TYPE:
          if ((setValue.s32 & 0x7FFFFFFF) == 0) predValue.u64 |= 1;
          break;
        case S64_TYPE:
          if ((setValue.s64 & 0x7FFFFFFFFFFFFFFF) == 0) predValue.u64 |= 1;
          break;
        case U8_TYPE:
        case B8_TYPE:
          if (setValue.u8 == 0) predValue.u64 |= 1;
          break;
        case U16_TYPE:
        case B16_TYPE:
          if (setValue.u16 == 0) predValue.u64 |= 1;
          break;
        case U32_TYPE:
        case B32_TYPE:
          if (setValue.u32 == 0) predValue.u64 |= 1;
          break;
        case U64_TYPE:
        case B64_TYPE:
          if (setValue.u64 == 0) predValue.u64 |= 1;
          break;
        case F16_TYPE:
          if (setValue.f16 == 0) predValue.u64 |= 1;
          break;
        case F32_TYPE:
          if (setValue.f32 == 0) predValue.u64 |= 1;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          if (setValue.f64 == 0) predValue.u64 |= 1;
          break;
        default:
          assert(0);
          break;
      }

      if ((type == S8_TYPE) || (type == S16_TYPE) || (type == S32_TYPE) ||
          (type == S64_TYPE) || (type == U8_TYPE) || (type == U16_TYPE) ||
          (type == U32_TYPE) || (type == U64_TYPE) || (type == B8_TYPE) ||
          (type == B16_TYPE) || (type == B32_TYPE) || (type == B64_TYPE)) {
        if ((setValue.u32 & (1 << (size - 1))) != 0) predValue.u64 |= 1 << 1;
      }
      if (type == F32_TYPE) {
        if (setValue.f32 < 0) predValue.u64 |= 1 << 1;
      }

      if (dst.get_operand_lohi() == 1) {
        setValue.u64 =
            ((m_regs.back()[regName].u64) & (~(0xFFFF))) + (data.u64 & 0xFFFF);
      } else if (dst.get_operand_lohi() == 2) {
        setValue.u64 = ((m_regs.back()[regName].u64) & (~(0xFFFF0000))) +
                       ((data.u64 << 16) & 0xFFFF0000);
      }

      set_reg(predName, predValue);
      set_reg(regName, setValue);
    } else if (type == BB128_TYPE) {
      // b128 stuff here.
      ptx_reg_t setValue2, setValue3, setValue4;
      setValue.u64 = 0;
      setValue2.u64 = 0;
      setValue3.u64 = 0;
      setValue4.u64 = 0;
      setValue.u32 = data.u128.lowest;
      setValue2.u32 = data.u128.low;
      setValue3.u32 = data.u128.high;
      setValue4.u32 = data.u128.highest;

      const symbol *name1, *name2, *name3, *name4 = NULL;

      name1 = dst.vec_symbol(0);
      name2 = dst.vec_symbol(1);
      name3 = dst.vec_symbol(2);
      name4 = dst.vec_symbol(3);

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
      set_reg(name3, setValue3);
      set_reg(name4, setValue4);
    } else if (type == BB64_TYPE || type == FF64_TYPE) {
      // ptxplus version of storing 64 bit values to registers stores to two
      // adjacent registers
      ptx_reg_t setValue2;
      setValue.u32 = 0;
      setValue2.u32 = 0;

      setValue.u32 = data.bits.ls;
      setValue2.u32 = data.bits.ms;

      const symbol *name1, *name2 = NULL;

      name1 = dst.vec_symbol(0);
      name2 = dst.vec_symbol(1);

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
    } else {
      if (dst.get_operand_lohi() == 1) {
        setValue.u64 = ((m_regs.back()[dst.get_symbol()].u64) & (~(0xFFFF))) +
                       (data.u64 & 0xFFFF);
      } else if (dst.get_operand_lohi() == 2) {
        setValue.u64 =
            ((m_regs.back()[dst.get_symbol()].u64) & (~(0xFFFF0000))) +
            ((data.u64 << 16) & 0xFFFF0000);
      }
      set_reg(dst.get_symbol(), setValue);
    }
  }

  // global memory - g[4], g[$r0]
  else if (dst.get_addr_space() == global_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = global_space;
  }

  // shared memory - s[4], s[$r0]
  else if (dst.get_addr_space() == shared_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->m_shared_mem;
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = shared_space;
  }

  // local memory - l0[4], l0[$r0]
  else if (dst.get_addr_space() == local_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->m_local_mem;
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = local_space;
  }

  else {
    printf("Destination stores to unknown location.");
    assert(0);
  }
}

void ptx_thread_info::set_vector_operand_values(const operand_info &dst,
                                                const ptx_reg_t &data1,
                                                const ptx_reg_t &data2,
                                                const ptx_reg_t &data3,
                                                const ptx_reg_t &data4) {
  unsigned num_elements = dst.get_vect_nelem();
  if (num_elements > 0) {
    set_reg(dst.vec_symbol(0), data1);
    if (num_elements > 1) {
      set_reg(dst.vec_symbol(1), data2);
      if (num_elements > 2) {
        set_reg(dst.vec_symbol(2), data3);
        if (num_elements > 3) {
          set_reg(dst.vec_symbol(3), data4);
        }
      }
    }
  }

  m_last_set_operand_value = data1;
}
void ptx_thread_info::set_wmma_vector_operand_values(
    const operand_info &dst, const ptx_reg_t &data1, const ptx_reg_t &data2,
    const ptx_reg_t &data3, const ptx_reg_t &data4, const ptx_reg_t &data5,
    const ptx_reg_t &data6, const ptx_reg_t &data7, const ptx_reg_t &data8) {
  unsigned num_elements = dst.get_vect_nelem();
  if (num_elements == 8) {
    set_reg(dst.vec_symbol(0), data1);
    set_reg(dst.vec_symbol(1), data2);
    set_reg(dst.vec_symbol(2), data3);
    set_reg(dst.vec_symbol(3), data4);
    set_reg(dst.vec_symbol(4), data5);
    set_reg(dst.vec_symbol(5), data6);
    set_reg(dst.vec_symbol(6), data7);
    set_reg(dst.vec_symbol(7), data8);
  } else {
    printf("error:set_wmma_vector_operands");
  }

  m_last_set_operand_value = data8;
}

#define my_abs(a) (((a) < 0) ? (-a) : (a))

#define MY_MAX_I(a, b) (a > b) ? a : b
#define MY_MAX_F(a, b) isNaN(a) ? b : isNaN(b) ? a : (a > b) ? a : b

#define MY_MIN_I(a, b) (a < b) ? a : b
#define MY_MIN_F(a, b) isNaN(a) ? b : isNaN(b) ? a : (a < b) ? a : b

#define MY_INC_I(a, b) (a >= b) ? 0 : a + 1
#define MY_DEC_I(a, b) ((a == 0) || (a > b)) ? b : a - 1

#define MY_CAS_I(a, b, c) (a == b) ? c : a

#define MY_EXCH(a, b) b

void abs_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case S16_TYPE:
      d.s16 = my_abs(a.s16);
      break;
    case S32_TYPE:
      d.s32 = my_abs(a.s32);
      break;
    case S64_TYPE:
      d.s64 = my_abs(a.s64);
      break;
    case U16_TYPE:
      d.s16 = my_abs(a.u16);
      break;
    case U32_TYPE:
      d.s32 = my_abs(a.u32);
      break;
    case U64_TYPE:
      d.s64 = my_abs(a.u64);
      break;
    case F32_TYPE:
      d.f32 = my_abs(a.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = my_abs(a.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void addp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // PTXPlus add instruction with carry (carry is kept in a predicate) register
  ptx_reg_t src1_data, src2_data, src3_data, data;
  int overflow = 0;
  int carry = 0;

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);

  unsigned rounding_mode = pI->rounding_mode();
  int orig_rm = fegetround();
  switch (rounding_mode) {
    case RN_OPTION:
      break;
    case RZ_OPTION:
      fesetround(FE_TOWARDZERO);
      break;
    default:
      assert(0);
      break;
  }

  // performs addition. Sets carry and overflow if needed.
  // src3_data.pred&0x4 is the carry flag
  switch (i_type) {
    case S8_TYPE:
      data.s64 = (src1_data.s64 & 0x0000000FF) + (src2_data.s64 & 0x0000000FF) +
                 (src3_data.pred & 0x4);
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) == 0) {
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000000100) >> 8;
      break;
    case S16_TYPE:
      data.s64 = (src1_data.s64 & 0x00000FFFF) + (src2_data.s64 & 0x00000FFFF) +
                 (src3_data.pred & 0x4);
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) == 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000010000) >> 16;
      break;
    case S32_TYPE:
      data.s64 = (src1_data.s64 & 0x0FFFFFFFF) + (src2_data.s64 & 0x0FFFFFFFF) +
                 (src3_data.pred & 0x4);
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) == 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case S64_TYPE:
      if (src2.is_literal()) {
        data.s64 = src1_data.s64 + src2_data.s32 + (src3_data.pred & 0x4);
      } else {
        data.s64 = src1_data.s64 + src2_data.s64 + (src3_data.pred & 0x4);
      }
      break;
    case U8_TYPE:
      data.u64 = (src1_data.u64 & 0xFF) + (src2_data.u64 & 0xFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x100) >> 8;
      break;
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) + (src2_data.u64 & 0xFFFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x10000) >> 16;
      break;
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) + (src2_data.u64 & 0xFFFFFFFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case U64_TYPE:
      data.s64 = src1_data.s64 + src2_data.s64 + (src3_data.pred & 0x4);
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 + src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 + src2_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 + src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }
  fesetround(orig_rm);

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);
}

bool print_debug_insts = false;

void add_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  //   if(thread->get_ctaid().x == 0 && thread->get_ctaid().y == 0 && thread->get_ctaid().z == 0)
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }

  ptx_reg_t src1_data, src2_data, data;
  int overflow = 0;
  int carry = 0;

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  unsigned rounding_mode = pI->rounding_mode();
  int orig_rm = fegetround();
  switch (rounding_mode) {
    case RN_OPTION:
      break;
    case RZ_OPTION:
      fesetround(FE_TOWARDZERO);
      break;
    default:
      assert(0);
      break;
  }

  // performs addition. Sets carry and overflow if needed.
  switch (i_type) {
    case S8_TYPE:
      data.s64 = (src1_data.s64 & 0x0000000FF) + (src2_data.s64 & 0x0000000FF);
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) == 0) {
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000000100) >> 8;
      break;
    case S16_TYPE:
      data.s64 = (src1_data.s64 & 0x00000FFFF) + (src2_data.s64 & 0x00000FFFF);
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) == 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000010000) >> 16;
      break;
    case S32_TYPE:
      data.s64 = (src1_data.s64 & 0x0FFFFFFFF) + (src2_data.s64 & 0x0FFFFFFFF);
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) == 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 + src2_data.s64;
      break;
    case U8_TYPE:
      data.u64 = (src1_data.u64 & 0xFF) + (src2_data.u64 & 0xFF);
      carry = (data.u64 & 0x100) >> 8;
      break;
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) + (src2_data.u64 & 0xFFFF);
      carry = (data.u64 & 0x10000) >> 16;
      break;
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) + (src2_data.u64 & 0xFFFFFFFF);
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 + src2_data.u64;
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 + src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 + src2_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 + src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }
  fesetround(orig_rm);

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);
}

void addc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void and_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(~(src1_data.pred) & ~(src2_data.pred));
  else
    data.u64 = src1_data.u64 & src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void andn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case B16_TYPE:
      src2_data.u16 = ~src2_data.u16;
      break;
    case B32_TYPE:
      src2_data.u32 = ~src2_data.u32;
      break;
    case B64_TYPE:
      src2_data.u64 = ~src2_data.u64;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  data.u64 = src1_data.u64 & src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void bar_callback(const inst_t *inst, ptx_thread_info *thread) {
  unsigned ctaid = thread->get_cta_uid();
  unsigned barid = inst->bar_id;
  unsigned value = thread->get_reduction_value(ctaid, barid);
  const ptx_instruction *pI = dynamic_cast<const ptx_instruction *>(inst);
  const operand_info &dst = pI->dst();
  ptx_reg_t data;
  data.u32 = value;
  thread->set_operand_value(dst, value, U32_TYPE, thread, pI);
}

void atom_callback(const inst_t *inst, ptx_thread_info *thread) {
  const ptx_instruction *pI = dynamic_cast<const ptx_instruction *>(inst);

  // "Decode" the output type
  unsigned to_type = pI->get_type();
  size_t size;
  int t;
  type_info_key::type_decode(to_type, size, t);

  // Set up operand variables
  ptx_reg_t data;       // d
  ptx_reg_t src1_data;  // a
  ptx_reg_t src2_data;  // b
  ptx_reg_t op_result;  // temp variable to hold operation result

  bool data_ready = false;

  // Get operand info of sources and destination
  const operand_info &dst = pI->dst();    // d
  const operand_info &src1 = pI->src1();  // a
  const operand_info &src2 = pI->src2();  // b

  // Get operand values
  src1_data = thread->get_operand_value(src1, src1, to_type, thread, 1);  // a
  if (dst.get_symbol()->type()) {
    src2_data = thread->get_operand_value(src2, dst, to_type, thread, 1);  // b
  } else {
    // This is the case whent he first argument (dest) is '_'
    src2_data = thread->get_operand_value(src2, src1, to_type, thread, 1);  // b
  }

  // Check state space
  addr_t effective_address = src1_data.u64;
  memory_space_t space = pI->get_space();
  if (space == undefined_space) {
    // generic space - determine space via address
    if (whichspace(effective_address) == global_space) {
      effective_address = generic_to_global(effective_address);
      space = global_space;
    } else if (whichspace(effective_address) == shared_space) {
      unsigned smid = thread->get_hw_sid();
      effective_address = generic_to_shared(smid, effective_address);
      space = shared_space;
    } else {
      abort();
    }
  }
  assert(space == global_space || space == shared_space);

  memory_space *mem = NULL;
  if (space == global_space)
    mem = thread->get_global_memory();
  else if (space == shared_space)
    mem = thread->m_shared_mem;
  else
    abort();

  // Copy value pointed to in operand 'a' into register 'd'
  // (i.e. copy src1_data to dst)
  mem->read(effective_address, size / 8, &data.s64);
  if (dst.get_symbol()->type()) {
    thread->set_operand_value(dst, data, to_type, thread,
                              pI);  // Write value into register 'd'
  }

  // Get the atomic operation to be performed
  unsigned m_atomic_spec = pI->get_atomic();

  switch (m_atomic_spec) {
    // AND
    case ATOMIC_AND: {
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 & src2_data.u32;
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 & src2_data.s32;
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.AND "
              "only accepts b32\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // OR
    case ATOMIC_OR: {
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 | src2_data.u32;
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 | src2_data.s32;
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.OR "
              "only accepts b32\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // XOR
    case ATOMIC_XOR: {
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 ^ src2_data.u32;
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 ^ src2_data.s32;
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.XOR "
              "only accepts b32\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // CAS
    case ATOMIC_CAS: {
      ptx_reg_t src3_data;
      const operand_info &src3 = pI->src3();
      src3_data = thread->get_operand_value(src3, dst, to_type, thread, 1);

      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = MY_CAS_I(data.u32, src2_data.u32, src3_data.u32);
          data_ready = true;
          break;
        case B64_TYPE:
        case U64_TYPE:
          op_result.u64 = MY_CAS_I(data.u64, src2_data.u64, src3_data.u64);
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_CAS_I(data.s32, src2_data.s32, src3_data.s32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.CAS "
              "only accepts b32 and b64\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // EXCH
    case ATOMIC_EXCH: {
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = MY_EXCH(data.u32, src2_data.u32);
          data_ready = true;
          break;
        case B64_TYPE:
        case U64_TYPE:
          op_result.u64 = MY_EXCH(data.u64, src2_data.u64);
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_EXCH(data.s32, src2_data.s32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.EXCH "
              "only accepts b32\n",
              to_type);
          assert(0);
          break;
      }

      break;
    }
      // ADD
    case ATOMIC_ADD: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = data.u32 + src2_data.u32;
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 + src2_data.s32;
          data_ready = true;
          break;
        case U64_TYPE:
          op_result.u64 = data.u64 + src2_data.u64;
          data_ready = true;
          break;
        case F32_TYPE:
          op_result.f32 = data.f32 + src2_data.f32;
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.ADD only "
              "accepts u32, s32, u64, and f32\n");
          assert(0);
          break;
      }

      break;
    }
      // INC
    case ATOMIC_INC: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_INC_I(data.u32, src2_data.u32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.INC only "
              "accepts u32 and s32\n");
          assert(0);
          break;
      }

      break;
    }
      // DEC
    case ATOMIC_DEC: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_DEC_I(data.u32, src2_data.u32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.DEC only "
              "accepts u32 and s32\n");
          assert(0);
          break;
      }

      break;
    }
      // MIN
    case ATOMIC_MIN: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_MIN_I(data.u32, src2_data.u32);
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_MIN_I(data.s32, src2_data.s32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.MIN only "
              "accepts u32 and s32\n");
          assert(0);
          break;
      }

      break;
    }
      // MAX
    case ATOMIC_MAX: {
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_MAX_I(data.u32, src2_data.u32);
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_MAX_I(data.s32, src2_data.s32);
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.MAX only "
              "accepts u32 and s32\n");
          assert(0);
          break;
      }

      break;
    }
      // DEFAULT
    default: {
      assert(0);
      break;
    }
  }

  // Write operation result into  memory
  // (i.e. copy src1_data to dst)
  if (data_ready) {
    mem->write(effective_address, size / 8, &op_result.s64, thread, pI);
  } else {
    printf("Execution error: data_ready not set\n");
    assert(0);
  }
}

// atom_impl will now result in a callback being called in mem_ctrl_pop
// (gpu-sim.c)
void atom_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // SYNTAX
  // atom.space.operation.type d, a, b[, c]; (now read in callback)

  // obtain memory space of the operation
  memory_space_t space = pI->get_space();

  // get the memory address
  const operand_info &src1 = pI->src1();
  // const operand_info &dst  = pI->dst();  // not needed for effective address
  // calculation
  unsigned i_type = pI->get_type();
  ptx_reg_t src1_data;
  src1_data = thread->get_operand_value(src1, src1, i_type, thread, 1);
  addr_t effective_address = src1_data.u64;

  addr_t effective_address_final;

  // handle generic memory space by converting it to global
  if (space == undefined_space) {
    if (whichspace(effective_address) == global_space) {
      effective_address_final = generic_to_global(effective_address);
      space = global_space;
    } else if (whichspace(effective_address) == shared_space) {
      unsigned smid = thread->get_hw_sid();
      effective_address_final = generic_to_shared(smid, effective_address);
      space = shared_space;
    } else {
      abort();
    }
  } else {
    assert(space == global_space || space == shared_space);
    effective_address_final = effective_address;
  }

  // Check state space
  assert(space == global_space || space == shared_space);

  thread->m_last_effective_address = effective_address_final;
  thread->m_last_memory_space = space;
  thread->m_last_dram_callback.function = atom_callback;
  thread->m_last_dram_callback.instruction = pI;
}

void bar_impl(const ptx_instruction *pIin, ptx_thread_info *thread) {
  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);
  unsigned bar_op = pI->barrier_op();
  unsigned red_op = pI->get_atomic();
  unsigned ctaid = thread->get_cta_uid();

  switch (bar_op) {
    case SYNC_OPTION: {
      if (pI->get_num_operands() > 1) {
        const operand_info &op0 = pI->dst();
        const operand_info &op1 = pI->src1();
        ptx_reg_t op0_data;
        ptx_reg_t op1_data;
        op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1);
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);
        pI->set_bar_id(op0_data.u32);
        pI->set_bar_count(op1_data.u32);
      } else {
        const operand_info &op0 = pI->dst();
        ptx_reg_t op0_data;
        op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1);
        pI->set_bar_id(op0_data.u32);
      }
      break;
    }
    case ARRIVE_OPTION: {
      const operand_info &op0 = pI->dst();
      const operand_info &op1 = pI->src1();
      ptx_reg_t op0_data;
      ptx_reg_t op1_data;
      op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1);
      op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);
      pI->set_bar_id(op0_data.u32);
      pI->set_bar_count(op1_data.u32);
      break;
    }
    case RED_OPTION: {
      if (pI->get_num_operands() > 3) {
        const operand_info &op1 = pI->src1();
        const operand_info &op2 = pI->src2();
        const operand_info &op3 = pI->src3();
        ptx_reg_t op1_data;
        ptx_reg_t op2_data;
        ptx_reg_t op3_data;
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);
        op2_data = thread->get_operand_value(op2, op2, U32_TYPE, thread, 1);
        op3_data = thread->get_operand_value(op3, op3, PRED_TYPE, thread, 1);
        op3_data.u32 = !(op3_data.pred & 0x0001);
        pI->set_bar_id(op1_data.u32);
        pI->set_bar_count(op2_data.u32);
        switch (red_op) {
          case ATOMIC_POPC:
            thread->popc_reduction(ctaid, op1_data.u32, op3_data.u32);
            break;
          case ATOMIC_AND:
            thread->and_reduction(ctaid, op1_data.u32, op3_data.u32);
            break;
          case ATOMIC_OR:
            thread->or_reduction(ctaid, op1_data.u32, op3_data.u32);
            break;
          default:
            abort();
            break;
        }
      } else {
        const operand_info &op1 = pI->src1();
        const operand_info &op2 = pI->src2();
        ptx_reg_t op1_data;
        ptx_reg_t op2_data;
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);
        op2_data = thread->get_operand_value(op2, op2, PRED_TYPE, thread, 1);
        op2_data.u32 = !(op2_data.pred & 0x0001);
        pI->set_bar_id(op1_data.u32);
        pI->set_bar_count(thread->get_ntid().x * thread->get_ntid().y *
                          thread->get_ntid().z);
        switch (red_op) {
          case ATOMIC_POPC:
            thread->popc_reduction(ctaid, op1_data.u32, op2_data.u32);
            break;
          case ATOMIC_AND:
            thread->and_reduction(ctaid, op1_data.u32, op2_data.u32);
            break;
          case ATOMIC_OR:
            thread->or_reduction(ctaid, op1_data.u32, op2_data.u32);
            break;
          default:
            abort();
            break;
        }
      }
      break;
    }
    default:
      abort();
      break;
  }

  thread->m_last_dram_callback.function = bar_callback;
  thread->m_last_dram_callback.instruction = pIin;
}


// TODO-LUCY: Check if this implementation matches AWARE
void bfe_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  unsigned i_type = pI->get_type();
  unsigned msb = (i_type == U32_TYPE || i_type == S32_TYPE) ? 31 : 63;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  ptx_reg_t src = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1);
  ptx_reg_t data;
  unsigned pos = b.u32 & 0xFF;
  unsigned len = c.u32 & 0xFF;
  switch (i_type) {
    case U32_TYPE: {
      unsigned mask;
      data.u32 = src.u32 >> pos;
      mask = 0xFFFFFFFF >> (32 - len);
      data.u32 &= mask;
      break;
    }
    case U64_TYPE: {
      unsigned long mask;
      data.u64 = src.u64 >> pos;
      mask = 0xFFFFFFFFFFFFFFFF >> (64 - len);
      data.u64 &= mask;
      break;
    }
    case S32_TYPE: {
      unsigned mask;
      unsigned min = MY_MIN_I(pos + len - 1, msb);
      unsigned sbit = len == 0 ? 0 : (src.s32 >> min) & 0x1;
      data.s32 = src.s32 >> pos;
      if (sbit > 0) {
        mask = 0xFFFFFFFF << len;
        data.s32 |= mask;
      } else {
        mask = 0xFFFFFFFF >> (32 - len);
        data.s32 &= mask;
      }
      break;
    }
    case S64_TYPE: {
      unsigned long mask;
      unsigned min = MY_MIN_I(pos + len - 1, msb);
      unsigned sbit = len == 0 ? 0 : (src.s64 >> min) & 0x1;
      data.s64 = src.s64 >> pos;
      if (sbit > 0) {
        mask = 0xFFFFFFFFFFFFFFFF << len;
        data.s64 |= mask;
      } else {
        mask = 0xFFFFFFFFFFFFFFFF >> (64 - len);
        data.s64 &= mask;
      }
      break;
    }
    default:
      printf("Operand type not supported for BFE instruction.\n");
      abort();
      return;
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void bfi_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  int i, max;
  ptx_reg_t src1_data, src2_data;
  ptx_reg_t src3_data, src4_data, data;

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  const operand_info &src4 = pI->src4();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);
  src4_data = thread->get_operand_value(src4, dst, i_type, thread, 1);

  switch (i_type) {
    case B32_TYPE:
      max = 32;
      break;
    case B64_TYPE:
      max = 64;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }
  data = src2_data;
  unsigned pos = src3_data.u32 & 0xFF;
  unsigned len = src4_data.u32 & 0xFF;
  for (i = 0; i < len && pos + i < max; i++) {
    data.u32 = (~((0x00000001) << (pos + i))) & data.u32;
    data.u32 = data.u32 | ((src1_data.u32 & ((0x00000001) << (i))) << (pos));
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);
}
void bfind_impl(const ptx_instruction *pI, ptx_thread_info *thread)
{
  const operand_info &dst  = pI->dst();
  const operand_info &src1 = pI->src1();
  const unsigned i_type = pI->get_type();

  const ptx_reg_t src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  const int msb = ( i_type == U32_TYPE || i_type == S32_TYPE) ? 31 : 63;

  unsigned long a = 0;
  switch (i_type)
  {
    case S32_TYPE: a = src1_data.s32; break;
    case U32_TYPE: a = src1_data.u32; break;
    case S64_TYPE: a = src1_data.s64; break;
    case U64_TYPE: a = src1_data.u64; break;
    default: assert(false); abort();
  }

  // negate negative signed inputs
  if ( ( i_type == S32_TYPE || i_type == S64_TYPE ) && ( a & ( 1 << msb ) ) ) {
      a = ~a;
  }
  uint32_t d_data = 0xffffffff;
  for (uint32_t i = msb; i >= 0; i--) {
      if (a & (1<<i))  { d_data = i; break; }
  }

  // if (.shiftamt && d != 0xffffffff)  { d = msb - d; }

  // store d
  thread->set_operand_value(dst, d_data, U32_TYPE, thread, pI);


}

void bra_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  // //   if(thread->get_ctaid().x == 2 && thread->get_ctaid().y == 89 && thread->get_ctaid().z == 0)
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }
  
  // if(pI->source_line() == 940)
  //   printf("this is where things actuallyn go wrong\n");
  
  const operand_info &target = pI->dst();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  thread->m_branch_taken = true;
  thread->set_npc(target_pc);
}

void brx_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = pI->dst();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  thread->m_branch_taken = true;
  thread->set_npc(target_pc);
}

void break_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = thread->pop_breakaddr();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  thread->m_branch_taken = true;
  thread->set_npc(target_pc);
}

void breakaddr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = pI->dst();
  thread->push_breakaddr(target);
  assert(
      pI->has_pred() ==
      false);  // pdom analysis cannot handle if this instruction is predicated
}

void brev_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);

  unsigned msb;
  switch (i_type) {
    case B32_TYPE:
      msb = 31;
      for (unsigned i = 0; i <= msb; i++) {
        if ((src1_data.u32 & (1 << i))) data.u32 |= 1 << (msb - i);
      }
      break;
    case B64_TYPE:
      msb = 63;
      for (unsigned i = 0; i <= msb; i++) {
        if ((src1_data.u64 & (1 << i))) data.u64 |= 1 << (msb - i);
      }
      break;
    default:
      assert(0);
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);
}
void brkpt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

unsigned trunc(unsigned num, unsigned precision) {
  int mask = 1, latest_one = -1;
  unsigned data = num;
  for (unsigned j = 0; j < sizeof(unsigned) * 8; j++) {
    int bit = data & mask;
    if (bit == 1) latest_one = j;
    data >>= 1;
  }
  if (latest_one >= precision) {
    // round_up is 1 if the most significant truncated digit is a 1, otherwise
    // it is 0
    // int round_up = (num & (1 << (latest_one-precision))) >>
    // (latest_one-precision); unsigned shifted_output = num >>
    // (latest_one-precision+1);
    // if shifted_output is a number like 1111, don't round up
    // if (shifted_output == (pow(2,precision)-1)) round_up = 0;
    // num = shifted_output + round_up;
    num >>= (latest_one - precision + 1);
  }
  return num;
}
void mapping(int thread, int wmma_type, int wmma_layout, int type, int index,
             int stride, int &row, int &col, int &assg_offset) {
  int offset;
  int c_row_offset[] = {0, 8, 0, 8, 4, 12, 4, 12};
  int c_col_offset[] = {0, 0, 8, 8, 0, 0, 8, 8};
  int c_tg_inside_row_offset[] = {0, 1, 0, 1};
  int c_tg_inside_col_offset[] = {0, 0, 2, 2};
  int c_inside_row_offset[] = {0, 0, 2, 2, 0, 0, 2, 2};
  int c_inside_col_offset[] = {0, 1, 0, 1, 4, 5, 4, 5};

  offset = thread_group_offset(thread, wmma_type, wmma_layout, type, stride);

  if (wmma_type == LOAD_A) {
    if (wmma_layout == ROW) {
      offset += index + 8 * ((thread % 16) / 8);
    } else {
      offset += 64 * (index / 4) + index % 4 + 128 * ((thread % 16) / 8);
    }
    offset = (offset / 16) * stride + offset % 16;
    assg_offset = index + 8 * ((thread % 16) / 8);
  } else if (wmma_type == LOAD_B) {
    if (wmma_layout == ROW) {
      offset += 64 * (index / 4) + index % 4 + 128 * ((thread % 16) / 8);
    } else {
      offset += index + 8 * ((thread % 16) / 8);
    }
    offset = (offset / 16) * stride + offset % 16;
    assg_offset = index + 8 * ((thread % 16) / 8);
  } else if (wmma_type == LOAD_C) {
    if (type == F16_TYPE) {
      row = c_row_offset[thread / 4] + thread % 4;
      col = c_col_offset[thread / 4] + index;
    } else {
      row = c_row_offset[thread / 4] + c_tg_inside_row_offset[thread % 4] +
            c_inside_row_offset[index];
      col = c_col_offset[thread / 4] + c_tg_inside_col_offset[thread % 4] +
            c_inside_col_offset[index];
    }
    assg_offset = index;
  }

  if (wmma_type == LOAD_A || wmma_type == LOAD_B) {
    if (wmma_layout == ROW) {
      row = offset / 16;
      col = offset % 16;
    } else {
      col = offset / 16;
      row = offset % 16;
    }
  }
}

void mma_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst) {
  int i, j, k, thrd;
  int row, col, offset;
  ptx_reg_t matrix_a[16][16];
  ptx_reg_t matrix_b[16][16];
  ptx_reg_t matrix_c[16][16];
  ptx_reg_t matrix_d[16][16];
  ptx_reg_t src_data;
  ptx_thread_info *thread;

  unsigned a_layout = pI->get_wmma_layout(0);
  unsigned b_layout = pI->get_wmma_layout(1);
  unsigned type = pI->get_type();
  unsigned type2 = pI->get_type2();
  int tid;
  const operand_info &dst = pI->operand_lookup(0);

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();
  float temp;
  half temp2;

  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    thread = core->get_thread_info()[tid + thrd];
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("THREAD=%d\n:", thrd);
    for (int operand_num = 1; operand_num <= 3; operand_num++) {
      const operand_info &src_a = pI->operand_lookup(operand_num);
      unsigned nelem = src_a.get_vect_nelem();
      ptx_reg_t v[8];
      thread->get_vector_operand_values(src_a, v, nelem);
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("Thread%d_Iteration=%d\n:", thrd, operand_num);
        for (k = 0; k < nelem; k++) {
          printf("%llx ", v[k].u64);
        }
        printf("\n");
      }
      ptx_reg_t nw_v[16];
      int hex_val;

      if (!((operand_num == 3) && (type2 == F32_TYPE))) {
        for (k = 0; k < 2 * nelem; k++) {
          if (k % 2 == 1)
            hex_val = (v[k / 2].s64 & 0xffff);
          else
            hex_val = ((v[k / 2].s64 & 0xffff0000) >> 16);
          nw_v[k].f16 = *((half *)&hex_val);
        }
      }
      if (!((operand_num == 3) && (type2 == F32_TYPE))) {
        for (k = 0; k < 2 * nelem; k++) {
          temp = nw_v[k].f16;
          if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
            printf("%.2f ", temp);
        }
        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) printf("\n");
      } else {
        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
          for (k = 0; k < 8; k++) {
            printf("%.2f ", v[k].f32);
          }
          printf("\n");
        }
      }
      switch (operand_num) {
        case 1:  // operand 1
          for (k = 0; k < 8; k++) {
            mapping(thrd, LOAD_A, a_layout, F16_TYPE, k, 16, row, col, offset);
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("A:thread=%d,row=%d,col=%d,offset=%d\n", thrd, row, col,
                     offset);
            matrix_a[row][col] = nw_v[offset];
          }
          break;
        case 2:  // operand 2
          for (k = 0; k < 8; k++) {
            mapping(thrd, LOAD_B, b_layout, F16_TYPE, k, 16, row, col, offset);
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("B:thread=%d,row=%d,col=%d,offset=%d\n", thrd, row, col,
                     offset);
            matrix_b[row][col] = nw_v[offset];
          }
          break;
        case 3:  // operand 3
          for (k = 0; k < 8; k++) {
            mapping(thrd, LOAD_C, ROW, type2, k, 16, row, col, offset);
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("C:thread=%d,row=%d,col=%d,offset=%d\n", thrd, row, col,
                     offset);
            if (type2 != F16_TYPE) {
              matrix_c[row][col] = v[offset];
            } else {
              matrix_c[row][col] = nw_v[offset];
            }
          }
          break;
        default:
          printf("Invalid Operand Index\n");
      }
    }
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) printf("\n");
  }
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("MATRIX_A\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        temp = matrix_a[i][j].f16;
        printf("%.2f ", temp);
      }
      printf("\n");
    }
    printf("MATRIX_B\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        temp = matrix_b[i][j].f16;
        printf("%.2f ", temp);
      }
      printf("\n");
    }
    printf("MATRIX_C\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        if (type2 == F16_TYPE) {
          temp = matrix_c[i][j].f16;
          printf("%.2f ", temp);
        } else
          printf("%.2f ", matrix_c[i][j].f32);
      }
      printf("\n");
    }
  }
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
      matrix_d[i][j].f16 = 0;
    }
  }

  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
      for (k = 0; k < 16; k++) {
        matrix_d[i][j].f16 =
            matrix_d[i][j].f16 + matrix_a[i][k].f16 * matrix_b[k][j].f16;
      }
      if ((type == F16_TYPE) && (type2 == F16_TYPE))
        matrix_d[i][j].f16 += matrix_c[i][j].f16;
      else if ((type == F32_TYPE) && (type2 == F16_TYPE)) {
        temp2 = matrix_d[i][j].f16 + matrix_c[i][j].f16;
        temp = temp2;
        matrix_d[i][j].f32 = temp;
      } else if ((type == F16_TYPE) && (type2 == F32_TYPE)) {
        temp = matrix_d[i][j].f16;
        temp += matrix_c[i][j].f32;
        matrix_d[i][j].f16 = half(temp);
      } else {
        temp = matrix_d[i][j].f16;
        temp += matrix_c[i][j].f32;
        matrix_d[i][j].f32 = temp;
      }
    }
  }
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    printf("MATRIX_D\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        if (type == F16_TYPE) {
          temp = matrix_d[i][j].f16;
          printf("%.2f ", temp);
        } else
          printf("%.2f ", matrix_d[i][j].f32);
      }
      printf("\n");
    }
  }
  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    int row_t[8];
    int col_t[8];
    for (k = 0; k < 8; k++) {
      mapping(thrd, LOAD_C, ROW, type, k, 16, row_t[k], col_t[k], offset);
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
        printf("mma:store:row:%d,col%d\n", row_t[k], col_t[k]);
    }
    thread = core->get_thread_info()[tid + thrd];

    if (type == F32_TYPE) {
      thread->set_wmma_vector_operand_values(
          dst, matrix_d[row_t[0]][col_t[0]], matrix_d[row_t[1]][col_t[1]],
          matrix_d[row_t[2]][col_t[2]], matrix_d[row_t[3]][col_t[3]],
          matrix_d[row_t[4]][col_t[4]], matrix_d[row_t[5]][col_t[5]],
          matrix_d[row_t[6]][col_t[6]], matrix_d[row_t[7]][col_t[7]]);

      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("thread%d:", thrd);
        for (k = 0; k < 8; k++) {
          printf("%.2f ", matrix_d[row_t[k]][col_t[k]].f32);
        }
        printf("\n");
      }
    } else if (type == F16_TYPE) {
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("thread%d:", thrd);
        for (k = 0; k < 8; k++) {
          temp = matrix_d[row_t[k]][col_t[k]].f16;
          printf("%.2f ", temp);
        }
        printf("\n");

        printf("thread%d:", thrd);
        for (k = 0; k < 8; k++) {
          printf("%x ", (unsigned int)matrix_d[row_t[k]][col_t[k]].f16);
        }
        printf("\n");
      }
      ptx_reg_t nw_data1, nw_data2, nw_data3, nw_data4;
      nw_data1.s64 = ((matrix_d[row_t[0]][col_t[0]].s64 & 0xffff)) |
                     ((matrix_d[row_t[1]][col_t[1]].s64 & 0xffff) << 16);
      nw_data2.s64 = ((matrix_d[row_t[2]][col_t[2]].s64 & 0xffff)) |
                     ((matrix_d[row_t[3]][col_t[3]].s64 & 0xffff) << 16);
      nw_data3.s64 = ((matrix_d[row_t[4]][col_t[4]].s64 & 0xffff)) |
                     ((matrix_d[row_t[5]][col_t[5]].s64 & 0xffff) << 16);
      nw_data4.s64 = ((matrix_d[row_t[6]][col_t[6]].s64 & 0xffff)) |
                     ((matrix_d[row_t[7]][col_t[7]].s64 & 0xffff) << 16);
      thread->set_vector_operand_values(dst, nw_data1, nw_data2, nw_data3,
                                        nw_data4);
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
        printf("thread%d=%llx,%llx,%llx,%llx", thrd, nw_data1.s64, nw_data2.s64,
               nw_data3.s64, nw_data4.s64);

    } else {
      printf("wmma:mma:wrong type\n");
      abort();
    }
  }
}

void call_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static unsigned call_uid_next = 1;

  const operand_info &target = pI->func_addr();
  assert(target.is_function_address());
  const symbol *func_addr = target.get_symbol();
  function_info *target_func = func_addr->get_pc();
  if (target_func->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           target_func->get_name().c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           target_func->get_name().c_str());
    /*
     * Some of the instructions like printf() gives the gpgpusim the wrong
     * impression that it is a function call. As printf() doesnt have a body
     * like functions do, doing pdom analysis for printf() causes a crash.
     */
    if (target_func->get_function_size() > 0) target_func->do_pdom();
    target_func->set_pdom();
  }

  // check that number of args and return match function requirements
  if (pI->has_return() ^ target_func->has_return()) {
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
        "between\n"
        "               call instruction and function declaration\n");
    abort();
  }
  unsigned n_return = target_func->has_return();
  unsigned n_args = target_func->num_args();
  unsigned n_operands = pI->get_num_operands();

  if (n_operands != (n_return + 1 + n_args)) {
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of arguements "
        "between\n"
        "               call instruction and function declaration\n");
    abort();
  }

  // handle intrinsic functions
  std::string fname = target_func->get_name();
  if (fname == "vprintf") {
    gpgpusim_cuda_vprintf(pI, thread, target_func);
    return;
  }
#if (CUDART_VERSION >= 5000)
  // Jin: handle device runtime apis for CDP
  else if (fname == "cudaGetParameterBufferV2") {
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_getParameterBufferV2(
        pI, thread, target_func);
    return;
  } else if (fname == "cudaLaunchDeviceV2") {
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_launchDeviceV2(
        pI, thread, target_func);
    return;
  } else if (fname == "cudaStreamCreateWithFlags") {
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_streamCreateWithFlags(
        pI, thread, target_func);
    return;
  }
#endif

  // read source arguements into register specified in declaration of function
  arg_buffer_list_t arg_values;
  copy_args_into_buffer_list(pI, thread, target_func, arg_values);

  // record local for return value (we only support a single return value)
  const symbol *return_var_src = NULL;
  const symbol *return_var_dst = NULL;
  if (target_func->has_return()) {
    return_var_dst = pI->dst().get_symbol();
    return_var_src = target_func->get_return_var();
  }

  gpgpu_sim *gpu = thread->get_gpu();
  unsigned callee_pc = 0, callee_rpc = 0;
  // if (gpu->simd_model() == POST_DOMINATOR) {
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc);
    assert(callee_pc == thread->get_pc());
  // }

  thread->callstack_push(callee_pc + pI->inst_size(), callee_rpc,
                         return_var_src, return_var_dst, call_uid_next++);

  copy_buffer_list_into_frame(thread, arg_values);

  thread->set_npc(target_func);
}

// Ptxplus version of call instruction. Jumps to a label not a different Kernel.
void callp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static unsigned call_uid_next = 1;

  const operand_info &target = pI->dst();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  const symbol *return_var_src = NULL;
  const symbol *return_var_dst = NULL;

  gpgpu_sim *gpu = thread->get_gpu();
  unsigned callee_pc = 0, callee_rpc = 0;
  if (gpu->simd_model() == POST_DOMINATOR) {
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc);
    assert(callee_pc == thread->get_pc());
  }

  thread->callstack_push_plus(callee_pc + pI->inst_size(), callee_rpc,
                              return_var_src, return_var_dst, call_uid_next++);
  thread->set_npc(target_pc);
}

void clz_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  int max;
  unsigned long long mask;
  d.u64 = 0;

  switch (i_type) {
    case B32_TYPE:
      max = 32;
      mask = 0x80000000;
      break;
    case B64_TYPE:
      max = 64;
      mask = 0x8000000000000000;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  while ((d.u32 < max) && ((a.u64 & mask) == 0)) {
    d.u32++;
    a.u64 = a.u64 << 1;
  }

  thread->set_operand_value(dst, d, B32_TYPE, thread, pI);
}

void cnot_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case PRED_TYPE:
      d.pred = ((a.pred & 0x0001) == 0) ? 1 : 0;
      break;
    case B16_TYPE:
      d.u16 = (a.u16 == 0) ? 1 : 0;
      break;
    case B32_TYPE:
      d.u32 = (a.u32 == 0) ? 1 : 0;
      break;
    case B64_TYPE:
      d.u64 = (a.u64 == 0) ? 1 : 0;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void cos_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      d.f32 = cos(a.f32);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

ptx_reg_t chop(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  switch (to_width) {
    case 8:
      x.mask_and(0, 0xFF);
      break;
    case 16:
      x.mask_and(0, 0xFFFF);
      break;
    case 32:
      x.mask_and(0, 0xFFFFFFFF);
      break;
    case 64:
      break;
    default:
      assert(0);
  }
  return x;
}

ptx_reg_t sext(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  x = chop(x, 0, from_width, 0, rounding_mode, saturation_mode);
  switch (from_width) {
    case 8:
      if (x.get_bit(7)) x.mask_or(0xFFFFFFFF, 0xFFFFFF00);
      break;
    case 16:
      if (x.get_bit(15)) x.mask_or(0xFFFFFFFF, 0xFFFF0000);
      break;
    case 32:
      if (x.get_bit(31)) x.mask_or(0xFFFFFFFF, 0x00000000);
      break;
    case 64:
      break;
    default:
      assert(0);
  }
  return x;
}

// sign extend depending on the destination register size - hack to get
// SobelFilter working in CUDA 4.2
ptx_reg_t sexd(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  x = chop(x, 0, from_width, 0, rounding_mode, saturation_mode);
  switch (to_width) {
    case 8:
      if (x.get_bit(7)) x.mask_or(0xFFFFFFFF, 0xFFFFFF00);
      break;
    case 16:
      if (x.get_bit(15)) x.mask_or(0xFFFFFFFF, 0xFFFF0000);
      break;
    case 32:
      if (x.get_bit(31)) x.mask_or(0xFFFFFFFF, 0x00000000);
      break;
    case 64:
      break;
    default:
      assert(0);
  }
  return x;
}

ptx_reg_t zext(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  return chop(x, 0, from_width, 0, rounding_mode, saturation_mode);
}

int saturatei(int a, int max, int min) {
  if (a > max)
    a = max;
  else if (a < min)
    a = min;
  return a;
}

unsigned int saturatei(unsigned int a, unsigned int max) {
  if (a > max) a = max;
  return a;
}

ptx_reg_t f2x(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  half mytemp;
  half_float::half tmp_h;
  // assert( from_width == 32);

  enum cudaRoundMode mode = cudaRoundZero;
  switch (rounding_mode) {
    case RZI_OPTION:
      mode = cudaRoundZero;
      break;
    case RNI_OPTION:
      mode = cudaRoundNearest;
      break;
    case RMI_OPTION:
      mode = cudaRoundMinInf;
      break;
    case RPI_OPTION:
      mode = cudaRoundPosInf;
      break;
    default:
      break;
  }

  ptx_reg_t y;
  if (to_sign == 1) {  // convert to 64-bit number first?
    int tmp = cuda_math::float2int(x.f32, mode);
    if ((x.u32 & 0x7f800000) == 0) tmp = 0;  // round denorm. FP to 0
    if (saturation_mode && to_width < 32) {
      tmp = saturatei(tmp, (1 << to_width) - 1, -(1 << to_width));
    }
    switch (to_width) {
      case 8:
        y.s8 = (char)tmp;
        break;
      case 16:
        y.s16 = (short)tmp;
        break;
      case 32:
        y.s32 = (int)tmp;
        break;
      case 64:
        y.s64 = (long long)tmp;
        break;
      default:
        assert(0);
        break;
    }
  } else if (to_sign == 0) {
    unsigned int tmp = cuda_math::float2uint(x.f32, mode);
    if ((x.u32 & 0x7f800000) == 0) tmp = 0;  // round denorm. FP to 0
    if (saturation_mode && to_width < 32) {
      tmp = saturatei(tmp, (1 << to_width) - 1);
    }
    switch (to_width) {
      case 8:
        y.u8 = (unsigned char)tmp;
        break;
      case 16:
        y.u16 = (unsigned short)tmp;
        break;
      case 32:
        y.u32 = (unsigned int)tmp;
        break;
      case 64:
        y.u64 = (unsigned long long)tmp;
        break;
      default:
        assert(0);
        break;
    }
  } else {
    switch (to_width) {
      case 16:
        y.f16 = half_float::half_cast<half,
                                      std::numeric_limits<float>::round_style>(
            x.f32);  // mytemp;
        break;
      case 32:
        y.f32 = float(x.f16);
        break;  // handled by f2f
      case 64:
        y.f64 = x.f32;
        break;
      default:
        assert(0);
        break;
    }
  }
  return y;
}

double saturated2i(double a, double max, double min) {
  if (a > max)
    a = max;
  else if (a < min)
    a = min;
  return a;
}

ptx_reg_t d2x(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  assert(from_width == 64);

  double tmp;
  switch (rounding_mode) {
    case RZI_OPTION:
      tmp = trunc(x.f64);
      break;
    case RNI_OPTION:
      tmp = nearbyint(x.f64);
      break;
    case RMI_OPTION:
      tmp = floor(x.f64);
      break;
    case RPI_OPTION:
      tmp = ceil(x.f64);
      break;
    default:
      tmp = x.f64;
      break;
  }

  ptx_reg_t y;
  if (to_sign == 1) {
    tmp = saturated2i(tmp, ((1 << (to_width - 1)) - 1), (1 << (to_width - 1)));
    switch (to_width) {
      case 8:
        y.s8 = (char)tmp;
        break;
      case 16:
        y.s16 = (short)tmp;
        break;
      case 32:
        y.s32 = (int)tmp;
        break;
      case 64:
        y.s64 = (long long)tmp;
        break;
      default:
        assert(0);
        break;
    }
  } else if (to_sign == 0) {
    tmp = saturated2i(tmp, ((1 << (to_width - 1)) - 1), 0);
    switch (to_width) {
      case 8:
        y.u8 = (unsigned char)tmp;
        break;
      case 16:
        y.u16 = (unsigned short)tmp;
        break;
      case 32:
        y.u32 = (unsigned int)tmp;
        break;
      case 64:
        y.u64 = (unsigned long long)tmp;
        break;
      default:
        assert(0);
        break;
    }
  } else {
    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        y.f32 = x.f64;
        break;
      case 64:
        y.f64 = x.f64;  // should be handled by d2d
        break;
      default:
        assert(0);
        break;
    }
  }
  return y;
}

ptx_reg_t s2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y;

  if (from_width < 64) {  // 32-bit conversion
    y = sext(x, from_width, 32, 0, rounding_mode, saturation_mode);

    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        switch (rounding_mode) {
          case RZ_OPTION:
            y.f32 = cuda_math::__int2float_rz(y.s32);
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__int2float_rn(y.s32);
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__int2float_rd(y.s32);
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__int2float_ru(y.s32);
            break;
          default:
            break;
        }
        break;
      case 64:
        y.f64 = y.s32;
        break;  // no rounding needed
      default:
        assert(0);
        break;
    }
  } else {
    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        switch (rounding_mode) {
          case RZ_OPTION:
            y.f32 = cuda_math::__ll2float_rz(y.s64);
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__ll2float_rn(y.s64);
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__ll2float_rd(y.s64);
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__ll2float_ru(y.s64);
            break;
          default:
            break;
        }
        break;
      case 64:
        y.f64 = y.s64;
        break;  // no internal implementation found
      default:
        assert(0);
        break;
    }
  }

  // saturating an integer to 1 or 0?
  return y;
}

ptx_reg_t u2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y;

  if (from_width < 64) {  // 32-bit conversion
    y = zext(x, from_width, 32, 0, rounding_mode, saturation_mode);

    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        switch (rounding_mode) {
          case RZ_OPTION:
            y.f32 = cuda_math::__uint2float_rz(y.u32);
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__uint2float_rn(y.u32);
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__uint2float_rd(y.u32);
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__uint2float_ru(y.u32);
            break;
          default:
            break;
        }
        break;
      case 64:
        y.f64 = y.u32;
        break;  // no rounding needed
      default:
        assert(0);
        break;
    }
  } else {
    switch (to_width) {
      case 16:
        assert(0);
        break;
      case 32:
        switch (rounding_mode) {
          case RZ_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64);
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64);
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64);
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64);
            break;
          default:
            break;
        }
        break;
      case 64:
        y.f64 = y.u64;
        break;  // no internal implementation found
      default:
        assert(0);
        break;
    }
  }

  // saturating an integer to 1 or 0?
  return y;
}

ptx_reg_t f2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y;
  if (from_width == 16) {
    half_float::detail::uint16 val = x.u16;
    y.f32 = half_float::detail::half2float<float>(val);
  } else {
    switch (rounding_mode) {
      case RZI_OPTION:
        y.f32 = truncf(x.f32);
        break;
      case RNI_OPTION:
#if CUDART_VERSION >= 3000
        y.f32 = nearbyintf(x.f32);
#else
        y.f32 = cuda_math::__internal_nearbyintf(x.f32);
#endif
        break;
      case RMI_OPTION:
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
        } else {
          y.f32 = floorf(x.f32);
        }
        break;
      case RPI_OPTION:
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
        } else {
          y.f32 = ceilf(x.f32);
        }
        break;
      default:
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
        } else {
          y.f32 = x.f32;
        }
        break;
    }
#if CUDART_VERSION >= 3000
    if (isnanf(y.f32))
#else
    if (cuda_math::__cuda___isnanf(y.f32))
#endif
    {
      y.u32 = 0x7fffffff;
    } else if (saturation_mode) {
      y.f32 = cuda_math::__saturatef(y.f32);
    }
  }

  return y;
}

ptx_reg_t d2d(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y;
  switch (rounding_mode) {
    case RZI_OPTION:
      y.f64 = trunc(x.f64);
      break;
    case RNI_OPTION:
#if CUDART_VERSION >= 3000
      y.f64 = nearbyint(x.f64);
#else
      y.f64 = cuda_math::__internal_nearbyintf(x.f64);
#endif
      break;
    case RMI_OPTION:
      y.f64 = floor(x.f64);
      break;
    case RPI_OPTION:
      y.f64 = ceil(x.f64);
      break;
    default:
      y.f64 = x.f64;
      break;
  }
  if (std::isnan(y.f64)) {
    y.u64 = 0xfff8000000000000ull;
  } else if (saturation_mode) {
    y.f64 = cuda_math::__saturatef(y.f64);
  }
  return y;
}

ptx_reg_t (*g_cvt_fn[11][11])(ptx_reg_t x, unsigned from_width,
                              unsigned to_width, int to_sign, int rounding_mode,
                              int saturation_mode) = {
    {NULL, sext, sext, sext, NULL, sext, sext, sext, s2f, s2f, s2f},
    {chop, NULL, sext, sext, chop, NULL, sext, sext, s2f, s2f, s2f},
    {chop, sexd, NULL, sext, chop, chop, NULL, sext, s2f, s2f, s2f},
    {chop, chop, chop, NULL, chop, chop, chop, NULL, s2f, s2f, s2f},
    {NULL, zext, zext, zext, NULL, zext, zext, zext, u2f, u2f, u2f},
    {chop, NULL, zext, zext, chop, NULL, zext, zext, u2f, u2f, u2f},
    {chop, chop, NULL, zext, chop, chop, NULL, zext, u2f, u2f, u2f},
    {chop, chop, chop, NULL, chop, chop, chop, NULL, u2f, u2f, u2f},
    {f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, NULL, f2f, f2x},
    {f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2f, f2x},
    {d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2d}};

void ptx_round(ptx_reg_t &data, int rounding_mode, int type) {
  if (rounding_mode == RN_OPTION) {
    return;
  }
  switch (rounding_mode) {
    case RZI_OPTION:
      switch (type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n");
          assert(0);
          break;
        case F16_TYPE:
          data.f16 = truncf(data.f16);
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = truncf(data.f32);
          break;
        case F64_TYPE:
        case FF64_TYPE:
          if (data.f64 < 0)
            data.f64 = ceil(data.f64);  // negative
          else
            data.f64 = floor(data.f64);  // positive
          break;
        default:
          assert(0);
          break;
      }
      break;
    case RNI_OPTION:
      switch (type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n");
          assert(0);
          break;
        case F16_TYPE:  // assert(0); break;
#if CUDART_VERSION >= 3000
          data.f16 = nearbyintf(data.f16);
#else
          data.f16 = cuda_math::__cuda_nearbyintf(data.f16);
#endif
          break;
        case F32_TYPE:
#if CUDART_VERSION >= 3000
          data.f32 = nearbyintf(data.f32);
#else
          data.f32 = cuda_math::__cuda_nearbyintf(data.f32);
#endif
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = round(data.f64);
          break;
        default:
          assert(0);
          break;
      }
      break;
    case RMI_OPTION:
      switch (type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n");
          assert(0);
          break;
        case F16_TYPE:
          data.f16 = floorf(data.f16);
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = floorf(data.f32);
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = floor(data.f64);
          break;
        default:
          assert(0);
          break;
      }
      break;
    case RPI_OPTION:
      switch (type) {
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n");
          assert(0);
          break;
        case F16_TYPE:
          data.f16 = ceilf(data.f16);
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = ceilf(data.f32);
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = ceil(data.f64);
          break;
        default:
          assert(0);
          break;
      }
      break;
    default:
      break;
  }

  if (type == F32_TYPE) {
#if CUDART_VERSION >= 3000
    if (isnanf(data.f32))
#else
    if (cuda_math::__cuda___isnanf(data.f32))
#endif
    {
      data.u32 = 0x7fffffff;
    }
  }
  if ((type == F64_TYPE) || (type == FF64_TYPE)) {
    if (std::isnan(data.f64)) {
      data.u64 = 0xfff8000000000000ull;
    }
  }
}

void ptx_saturate(ptx_reg_t &data, int saturation_mode, int type) {
  if (!saturation_mode) {
    return;
  }
  switch (type) {
    case S8_TYPE:
    case S16_TYPE:
    case S32_TYPE:
    case S64_TYPE:
    case U8_TYPE:
    case U16_TYPE:
    case U32_TYPE:
    case U64_TYPE:
      printf("Trying to clamp an integer to 1??\n");
      assert(0);
      break;
    case F16_TYPE:                           // assert(0); break;
      if (data.f16 > 1.0f) data.f16 = 1.0f;  // negative
      if (data.f16 < 0.0f) data.f16 = 0.0f;  // positive
      break;
    case F32_TYPE:
      if (data.f32 > 1.0f) data.f32 = 1.0f;  // negative
      if (data.f32 < 0.0f) data.f32 = 0.0f;  // positive
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (data.f64 > 1.0f) data.f64 = 1.0f;  // negative
      if (data.f64 < 0.0f) data.f64 = 0.0f;  // positive
      break;
    default:
      assert(0);
      break;
  }
}

void cvt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  unsigned to_type = pI->get_type();
  unsigned from_type = pI->get_type2();
  unsigned rounding_mode = pI->rounding_mode();
  unsigned saturation_mode = pI->saturation_mode();

  //   if ( to_type == F16_TYPE || from_type == F16_TYPE )
  //      abort();

  int to_sign, from_sign;
  size_t from_width, to_width;
  unsigned src_fmt =
      type_info_key::type_decode(from_type, from_width, from_sign);
  unsigned dst_fmt = type_info_key::type_decode(to_type, to_width, to_sign);

  ptx_reg_t data = thread->get_operand_value(src1, dst, from_type, thread, 1);

  if (pI->is_neg()) {
    switch (from_type) {
      // Default to f32 for now, need to add support for others
      case S8_TYPE:
      case U8_TYPE:
      case B8_TYPE:
        data.s8 = -data.s8;
        break;
      case S16_TYPE:
      case U16_TYPE:
      case B16_TYPE:
        data.s16 = -data.s16;
        break;
      case S32_TYPE:
      case U32_TYPE:
      case B32_TYPE:
        data.s32 = -data.s32;
        break;
      case S64_TYPE:
      case U64_TYPE:
      case B64_TYPE:
        data.s64 = -data.s64;
        break;
      case F16_TYPE:
        data.f16 = -data.f16;
        break;
      case F32_TYPE:
        data.f32 = -data.f32;
        break;
      case F64_TYPE:
      case FF64_TYPE:
        data.f64 = -data.f64;
        break;
      default:
        assert(0);
    }
  }

  if (g_cvt_fn[src_fmt][dst_fmt] != NULL) {
    ptx_reg_t result = g_cvt_fn[src_fmt][dst_fmt](
        data, from_width, to_width, to_sign, rounding_mode, saturation_mode);
    data = result;
  }

  thread->set_operand_value(dst, data, to_type, thread, pI);
}

void cvta_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  memory_space_t space = pI->get_space();
  bool to_non_generic = pI->is_to();

  unsigned i_type = pI->get_type();
  ptx_reg_t from_addr = thread->get_operand_value(src1, dst, i_type, thread, 1);
  addr_t from_addr_hw = (addr_t)from_addr.u64;
  addr_t to_addr_hw = 0;
  unsigned smid = thread->get_hw_sid();
  unsigned hwtid = thread->get_hw_tid();

  if (to_non_generic) {
    switch (space.get_type()) {
      case shared_space:
        to_addr_hw = generic_to_shared(smid, from_addr_hw);
        break;
      case local_space:
        to_addr_hw = generic_to_local(smid, hwtid, from_addr_hw);
        break;
      case global_space:
        to_addr_hw = generic_to_global(from_addr_hw);
        break;
      default:
        abort();
    }
  } else {
    switch (space.get_type()) {
      case shared_space:
        to_addr_hw = shared_to_generic(smid, from_addr_hw);
        break;
      case local_space:
        to_addr_hw = local_to_generic(smid, hwtid, from_addr_hw) +
                     thread->get_local_mem_stack_pointer();
        break;  // add stack ptr here so that it can be passed as a pointer at
                // function call
      case global_space:
        to_addr_hw = global_to_generic(from_addr_hw);
        break;
      default:
        abort();
    }
  }

  ptx_reg_t to_addr;
  to_addr.u64 = to_addr_hw;
  thread->set_reg(dst.get_symbol(), to_addr);
}

void div_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();

  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case S8_TYPE:
      data.s8 = src1_data.s8 / src2_data.s8;
      break;
    case S16_TYPE:
      data.s16 = src1_data.s16 / src2_data.s16;
      break;
    case S32_TYPE:
      data.s32 = src1_data.s32 / src2_data.s32;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 / src2_data.s64;
      break;
    case U8_TYPE:
      data.u8 = src1_data.u8 / src2_data.u8;
      break;
    case U16_TYPE:
      data.u16 = src1_data.u16 / src2_data.u16;
      break;
    case U32_TYPE:
      data.u32 = src1_data.u32 / src2_data.u32;
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 / src2_data.u64;
      break;
    case B8_TYPE:
      data.u8 = src1_data.u8 / src2_data.u8;
      break;
    case B16_TYPE:
      data.u16 = src1_data.u16 / src2_data.u16;
      break;
    case B32_TYPE:
      data.u32 = src1_data.u32 / src2_data.u32;
      break;
    case B64_TYPE:
      data.u64 = src1_data.u64 / src2_data.u64;
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 / src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 / src2_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 / src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void dp4a_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  printf("DP4A instruction not implemented yet");
  assert(0);
}

void ex2_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();

  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      data.f32 = cuda_math::__powf(2.0, src1_data.f32);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void exit_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  thread->set_done();
  thread->exitCore();
  thread->registerExit();
}

void mad_def(const ptx_instruction *pI, ptx_thread_info *thread,
             bool use_carry = false);

void fma_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread);
}

void isspacep_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a;
  bool t = false;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  memory_space_t space = pI->get_space();

  a = thread->get_reg(src1.get_symbol());
  addr_t addr = (addr_t)a.u64;
  unsigned smid = thread->get_hw_sid();
  unsigned hwtid = thread->get_hw_tid();

  switch (space.get_type()) {
    case shared_space:
      t = isspace_shared(smid, addr);
    case local_space:
      t = isspace_local(smid, hwtid, addr);
    case global_space:
      t = isspace_global(addr);
    default:
      abort();
  }

  ptx_reg_t p;
  p.pred = t ? 1 : 0;

  thread->set_reg(dst.get_symbol(), p);
}

void decode_space(memory_space_t &space, ptx_thread_info *thread,
                  const operand_info &op, memory_space *&mem, addr_t &addr) {
  unsigned smid = thread->get_hw_sid();
  unsigned hwtid = thread->get_hw_tid();

  if (space == param_space_unclassified) {
    // need to op to determine whether it refers to a kernel param or local
    // param
    const symbol *s = op.get_symbol();
    const type_info *t = s->type();
    type_info_key ti = t->get_key();
    if (ti.is_param_kernel())
      space = param_space_kernel;
    else if (ti.is_param_local()) {
      space = param_space_local;
    }
    // mov r1, param-label
    else if (ti.is_reg()) {
      space = param_space_kernel;
    } else {
      printf("GPGPU-Sim PTX: ERROR ** cannot resolve .param space for '%s'\n",
             s->name().c_str());
      abort();
    }
  }
  switch (space.get_type()) {
    case global_space:
      mem = thread->get_global_memory();
      break;
    case param_space_local:
    case local_space:
      mem = thread->m_local_mem;
      addr += thread->get_local_mem_stack_pointer();
      break;
    case tex_space:
      mem = thread->get_tex_memory();
      break;
    case surf_space:
      mem = thread->get_surf_memory();
      break;
    case param_space_kernel:
      mem = thread->get_param_memory();
      break;
    case shared_space:
      mem = thread->m_shared_mem;
      break;
    case sstarr_space:
      mem = thread->m_sstarr_mem;
      break;
    case const_space:
      mem = thread->get_global_memory();
      break;
    case generic_space:
      if (thread->get_ptx_version().ver() >= 2.0) {
        // convert generic address to memory space address
        space = whichspace(addr);
        switch (space.get_type()) {
          case global_space:
            mem = thread->get_global_memory();
            addr = generic_to_global(addr);
            break;
          case local_space:
            mem = thread->m_local_mem;
            addr = generic_to_local(smid, hwtid, addr);
            break;
          case shared_space:
            mem = thread->m_shared_mem;
            addr = generic_to_shared(smid, addr);
            break;
          default:
            abort();
        }
      } else {
        abort();
      }
      break;
    case param_space_unclassified:
    case undefined_space:
    default:
      abort();
  }
}

void ld_exec(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  // //   if(thread->get_ctaid().x == 2 && thread->get_ctaid().y == 89 && thread->get_ctaid().z == 0)
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned type = pI->get_type();

  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, type, thread, 1);
  ptx_reg_t data;
  memory_space_t space = pI->get_space();
  unsigned vector_spec = pI->get_vector();

  memory_space *mem = NULL;
  // addr_t addr = src1_data.u32;
  addr_t addr = src1_data.u64;

  decode_space(space, thread, src1, mem, addr);

  size_t size;
  int t;
  data.u64 = 0;
  type_info_key::type_decode(type, size, t);
  if (!vector_spec) {
    mem->read(addr, size / 8, &data.s64); // MRS_TODO: this is the correct one needed
    // memcpy(&(data.s64), addr64, size / 8);
    // printf("float value = %f\n", *((float*)addr64));
    if (type == S16_TYPE || type == S32_TYPE) sign_extend(data, size, dst);
    thread->set_operand_value(dst, data, type, thread, pI);
  } else {
    assert(0); //MRS_TODO: what happends here? turn this to 64 bit as well
    ptx_reg_t data1, data2, data3, data4;
    mem->read(addr, size / 8, &data1.s64);
    mem->read(addr + size / 8, size / 8, &data2.s64);
    if (vector_spec != V2_TYPE) {  // either V3 or V4
      mem->read(addr + 2 * size / 8, size / 8, &data3.s64);
      if (vector_spec != V3_TYPE) {  // v4
        mem->read(addr + 3 * size / 8, size / 8, &data4.s64);
        thread->set_vector_operand_values(dst, data1, data2, data3, data4);
      } else  // v3
        thread->set_vector_operand_values(dst, data1, data2, data3, data3);
    } else  // v2
      thread->set_vector_operand_values(dst, data1, data2, data2, data2);
  }
  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = space;
}

void ld_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ld_exec(pI, thread);
}
void ldu_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ld_exec(pI, thread);
}

void mma_st_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  size_t size;
  unsigned smid;
  int t;
  int thrd, k;
  ptx_thread_info *thread;

  const operand_info &src = pI->operand_lookup(1);
  const operand_info &src1 = pI->operand_lookup(0);
  const operand_info &src2 = pI->operand_lookup(2);
  int tid;
  unsigned type = pI->get_type();
  unsigned wmma_type = pI->get_wmma_type();
  unsigned wmma_layout = pI->get_wmma_layout(0);
  int stride;

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  _memory_op_t insn_memory_op =
      pI->has_memory_read() ? memory_load : memory_store;
  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    thread = core->get_thread_info()[tid + thrd];
    ptx_reg_t addr_reg = thread->get_operand_value(src1, src, type, thread, 1);
    ptx_reg_t src2_data = thread->get_operand_value(src2, src, type, thread, 1);
    const operand_info &src_a = pI->operand_lookup(1);
    unsigned nelem = src_a.get_vect_nelem();
    ptx_reg_t *v = new ptx_reg_t[8];
    thread->get_vector_operand_values(src_a, v, nelem);
    stride = src2_data.u32;

    memory_space_t space = pI->get_space();

    memory_space *mem = NULL;
    addr_t addr = addr_reg.u32;
    assert(0); // 32 bit address

    new_addr_type mem_txn_addr[MAX_ACCESSES_PER_INSN_PER_THREAD];
    int num_mem_txn = 0;

    smid = thread->get_hw_sid();
    if (whichspace(addr) == shared_space) {
      addr = generic_to_shared(smid, addr);
      space = shared_space;
    }
    decode_space(space, thread, src1, mem, addr);

    type_info_key::type_decode(type, size, t);
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("mma_st: thrd=%d, addr=%x, fp(size=%zu), stride=%d\n", thrd,
             addr_reg.u32, size, src2_data.u32);
    addr_t new_addr =
        addr + thread_group_offset(thrd, wmma_type, wmma_layout, type, stride) *
                   size / 8;
    addr_t push_addr;

    ptx_reg_t nw_v[8];
    for (k = 0; k < 8; k++) {
      if (k % 2 == 0)
        nw_v[k].s64 = (v[k / 2].s64 & 0xffff);
      else
        nw_v[k].s64 = ((v[k / 2].s64 & 0xffff0000) >> 16);
    }

    for (k = 0; k < 8; k++) {
      if (type == F32_TYPE) {
        // mem->write(new_addr+4*acc_float_offset(k,wmma_layout,stride),size/8,&v[k].s64,thread,pI);
        push_addr = new_addr + 4 * acc_float_offset(k, wmma_layout, stride);
        mem->write(push_addr, size / 8, &v[k].s64, thread, pI);
        mem_txn_addr[num_mem_txn++] = push_addr;

        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
          printf(
              "wmma:store:thread%d=%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n",
              thrd, v[0].s64, v[1].s64, v[2].s64, v[3].s64, v[4].s64, v[5].s64,
              v[6].s64, v[7].s64);
          float temp;
          int l;
          printf("thread=%d:", thrd);
          for (l = 0; l < 8; l++) {
            temp = v[l].f32;
            printf("%.2f", temp);
          }
          printf("\n");
        }
      } else if (type == F16_TYPE) {
        if (wmma_layout == ROW) {
          // mem->write(new_addr+k*2,size/8,&nw_v[k].s64,thread,pI);
          push_addr = new_addr + k * 2;
          mem->write(push_addr, size / 8, &nw_v[k].s64, thread, pI);
          if (k % 2 == 0) mem_txn_addr[num_mem_txn++] = push_addr;
        } else if (wmma_layout == COL) {
          // mem->write(new_addr+k*2*stride,size/8,&nw_v[k].s64,thread,pI);
          push_addr = new_addr + k * 2 * stride;
          mem->write(push_addr, size / 8, &nw_v[k].s64, thread, pI);
          mem_txn_addr[num_mem_txn++] = push_addr;
        }

        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
          printf(
              "wmma:store:thread%d=%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n",
              thrd, nw_v[0].s64, nw_v[1].s64, nw_v[2].s64, nw_v[3].s64,
              nw_v[4].s64, nw_v[5].s64, nw_v[6].s64, nw_v[7].s64);
      }
    }

    delete[] v;
    inst.space = space;
    inst.set_addr(thrd, (new_addr_type *)mem_txn_addr, num_mem_txn);

    if ((type == F16_TYPE) &&
        (wmma_layout == COL))  // check the profiling xls for details
      inst.data_size = 2;      // 2 byte transaction
    else
      inst.data_size = 4;  // 4 byte transaction

    assert(inst.memory_op == insn_memory_op);
    // thread->m_last_effective_address = addr;
    // thread->m_last_memory_space = space;
  }
}

void mma_ld_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  size_t size;
  int t, i;
  unsigned smid;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned type = pI->get_type();
  unsigned wmma_type = pI->get_wmma_type();
  unsigned wmma_layout = pI->get_wmma_layout(0);
  int tid;
  int thrd, stride;
  ptx_thread_info *thread;

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  _memory_op_t insn_memory_op =
      pI->has_memory_read() ? memory_load : memory_store;

  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    thread = core->get_thread_info()[tid + thrd];
    ptx_reg_t src1_data =
        thread->get_operand_value(src1, dst, U32_TYPE, thread, 1);
    ptx_reg_t src2_data =
        thread->get_operand_value(src2, dst, U32_TYPE, thread, 1);
    stride = src2_data.u32;
    memory_space_t space = pI->get_space();

    memory_space *mem = NULL;
    addr_t addr = src1_data.u32;
    assert(0); // 32 bit address
    smid = thread->get_hw_sid();
    if (whichspace(addr) == shared_space) {
      addr = generic_to_shared(smid, addr);
      space = shared_space;
    }

    decode_space(space, thread, src1, mem, addr);
    type_info_key::type_decode(type, size, t);

    ptx_reg_t data[16];
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("mma_ld: thrd=%d,addr=%x, fpsize=%zu, stride=%d\n", thrd,
             src1_data.u32, size, src2_data.u32);

    addr_t new_addr =
        addr + thread_group_offset(thrd, wmma_type, wmma_layout, type, stride) *
                   size / 8;
    addr_t fetch_addr;
    new_addr_type mem_txn_addr[MAX_ACCESSES_PER_INSN_PER_THREAD];
    int num_mem_txn = 0;

    if (wmma_type == LOAD_A) {
      for (i = 0; i < 16; i++) {
        if (wmma_layout == ROW) {
          // mem->read(new_addr+2*i,size/8,&data[i].s64);
          fetch_addr = new_addr + 2 * i;
          mem->read(fetch_addr, size / 8, &data[i].s64);
        } else if (wmma_layout == COL) {
          // mem->read(new_addr+2*(i%4)+2*stride*4*(i/4),size/8,&data[i].s64);
          fetch_addr = new_addr + 2 * (i % 4) + 2 * stride * 4 * (i / 4);
          mem->read(fetch_addr, size / 8, &data[i].s64);
        } else {
          printf("mma_ld:wrong_layout_type\n");
          abort();
        }
        if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr;
      }
    } else if (wmma_type == LOAD_B) {
      for (i = 0; i < 16; i++) {
        if (wmma_layout == COL) {
          // mem->read(new_addr+2*i,size/8,&data[i].s64);
          fetch_addr = new_addr + 2 * i;
          mem->read(fetch_addr, size / 8, &data[i].s64);
        } else if (wmma_layout == ROW) {
          // mem->read(new_addr+2*(i%4)+2*stride*4*(i/4),size/8,&data[i].s64);
          fetch_addr = new_addr + 2 * (i % 4) + 2 * stride * 4 * (i / 4);
          mem->read(fetch_addr, size / 8, &data[i].s64);
        } else {
          printf("mma_ld:wrong_layout_type\n");
          abort();
        }
        if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr;
      }
    } else if (wmma_type == LOAD_C) {
      for (i = 0; i < 8; i++) {
        if (type == F16_TYPE) {
          if (wmma_layout == ROW) {
            // mem->read(new_addr+2*i,size/8,&data[i].s64);
            fetch_addr = new_addr + 2 * i;
            mem->read(fetch_addr, size / 8, &data[i].s64);
            if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr;
          } else if (wmma_layout == COL) {
            // mem->read(new_addr+2*stride*i,size/8,&data[i].s64);
            fetch_addr = new_addr + 2 * stride * i;
            mem->read(fetch_addr, size / 8, &data[i].s64);
            mem_txn_addr[num_mem_txn++] = fetch_addr;
          } else {
            printf("mma_ld:wrong_type\n");
            abort();
          }
        } else if (type == F32_TYPE) {
          // mem->read(new_addr+4*acc_float_offset(i,wmma_layout,stride),size/8,&data[i].s64);
          fetch_addr = new_addr + 4 * acc_float_offset(i, wmma_layout, stride);
          mem->read(fetch_addr, size / 8, &data[i].s64);
          mem_txn_addr[num_mem_txn++] = fetch_addr;
        } else {
          printf("wrong type");
          abort();
        }
      }
    } else {
      printf("wrong wmma type\n");
      ;
      abort();
    }
    // generate timing memory request
    inst.space = space;
    inst.set_addr(thrd, (new_addr_type *)mem_txn_addr, num_mem_txn);

    if ((wmma_type == LOAD_C) && (type == F16_TYPE) &&
        (wmma_layout == COL))  // memory address is scattered, check the
                               // profiling xls for more detail.
      inst.data_size = 2;      // 2 byte transaction
    else
      inst.data_size = 4;  // 4 byte transaction
    assert(inst.memory_op == insn_memory_op);

    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
      if (type == F16_TYPE) {
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 16; i++) {
          printf("%llx ", data[i].u64);
        }
        printf("\n");

        printf("\nmma_ld:thread%d= ", thrd);
        float temp;
        for (i = 0; i < 16; i++) {
          temp = data[i].f16;
          printf("%.2f ", temp);
        }
        printf("\n");
      } else {
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 8; i++) {
          printf("%.2f ", data[i].f32);
        }
        printf("\n");
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 8; i++) {
          printf("%llx ", data[i].u64);
        }
        printf("\n");
      }
    }

    if ((wmma_type == LOAD_C) && (type == F32_TYPE)) {
      thread->set_wmma_vector_operand_values(dst, data[0], data[1], data[2],
                                             data[3], data[4], data[5], data[6],
                                             data[7]);
    } else {
      ptx_reg_t nw_data[8];
      int num_reg;

      if (wmma_type == LOAD_C)
        num_reg = 4;
      else
        num_reg = 8;

      for (i = 0; i < num_reg; i++) {
        nw_data[i].s64 = ((data[2 * i].s64 & 0xffff) << 16) |
                         ((data[2 * i + 1].s64 & 0xffff));
      }

      if (wmma_type == LOAD_C)
        thread->set_vector_operand_values(dst, nw_data[0], nw_data[1],
                                          nw_data[2], nw_data[3]);
      else
        thread->set_wmma_vector_operand_values(
            dst, nw_data[0], nw_data[1], nw_data[2], nw_data[3], nw_data[4],
            nw_data[5], nw_data[6], nw_data[7]);
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf(
            "mma_ld:data[0].s64=%llx,data[1].s64=%llx,new_data[0].s64=%llx\n",
            data[0].u64, data[1].u64, nw_data[0].u64);
        printf(
            "mma_ld:data[2].s64=%llx,data[3].s64=%llx,new_data[1].s64=%llx\n",
            data[2].u64, data[3].u64, nw_data[1].u64);
        printf(
            "mma_ld:data[4].s64=%llx,data[5].s64=%llx,new_data[2].s64=%llx\n",
            data[4].u64, data[5].u64, nw_data[2].u64);
        printf(
            "mma_ld:data[6].s64=%llx,data[7].s64=%llx,new_data[3].s64=%llx\n",
            data[6].u64, data[7].u64, nw_data[3].u64);
        if (wmma_type != LOAD_C) {
          printf(
              "mma_ld:data[8].s64=%llx,data[9].s64=%llx,new_data[4].s64=%llx\n",
              data[8].u64, data[9].u64, nw_data[4].s64);
          printf(
              "mma_ld:data[10].s64=%llx,data[11].s64=%llx,new_data[5].s64=%"
              "llx\n",
              data[10].u64, data[11].u64, nw_data[5].u64);
          printf(
              "mma_ld:data[12].s64=%llx,data[13].s64=%llx,new_data[6].s64=%"
              "llx\n",
              data[12].u64, data[13].u64, nw_data[6].u64);
          printf(
              "mma_ld:data[14].s64=%llx,data[15].s64=%llx,new_data[7].s64=%"
              "llx\n",
              data[14].u64, data[15].u64, nw_data[3].u64);
        }
      }
    }

    // thread->m_last_effective_address = addr;
    // thread->m_last_memory_space = space;
  }
}

void lg2_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();

  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      d.f32 = log(a.f32) / log(2);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void mad24_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  ptx_reg_t d, t;

  unsigned i_type = pI->get_type();
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  unsigned sat_mode = pI->saturation_mode();

  assert(!pI->is_wide());

  switch (i_type) {
    case S32_TYPE:
      t.s64 = a.s32 * b.s32;
      if (pI->is_hi()) {
        d.s64 = (t.s64 >> 16) + c.s32;
        if (sat_mode) {
          if (d.s64 > (int)0x7FFFFFFF)
            d.s64 = (int)0x7FFFFFFF;
          else if (d.s64 < (int)0x80000000)
            d.s64 = (int)0x80000000;
        }
      } else if (pI->is_lo())
        d.s64 = t.s32 + c.s32;
      else
        assert(0);
      break;
    case U32_TYPE:
      t.u64 = a.u32 * b.u32;
      if (pI->is_hi())
        d.u64 = (t.u64 >> 16) + c.u32;
      else if (pI->is_lo())
        d.u64 = t.u32 + c.u32;
      else
        assert(0);
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void mad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, false);
}

void madp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, true);
}

void madc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, true);
}

void mad_def(const ptx_instruction *pI, ptx_thread_info *thread,
             bool use_carry) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  ptx_reg_t d, t;

  int carry = 0;
  int overflow = 0;

  unsigned i_type = pI->get_type();
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  // take the carry bit, it should be the 4th operand
  ptx_reg_t carry_bit;
  carry_bit.u64 = 0;
  if (use_carry) {
    const operand_info &carry = pI->operand_lookup(4);
    carry_bit = thread->get_operand_value(carry, dst, PRED_TYPE, thread, 0);
    carry_bit.pred &= 0x4;
    carry_bit.pred >>= 2;
  }

  unsigned rounding_mode = pI->rounding_mode();

  switch (i_type) {
    case S16_TYPE:
      t.s32 = a.s16 * b.s16;
      if (pI->is_wide())
        d.s32 = t.s32 + c.s32 + carry_bit.pred;
      else if (pI->is_hi())
        d.s16 = (t.s32 >> 16) + c.s16 + carry_bit.pred;
      else if (pI->is_lo())
        d.s16 = t.s16 + c.s16 + carry_bit.pred;
      else
        assert(0);
      carry =
          ((long long int)(t.s32 + c.s32 + carry_bit.pred) & 0x100000000) >> 32;
      break;
    case S32_TYPE:
      t.s64 = a.s32 * b.s32;
      if (pI->is_wide())
        d.s64 = t.s64 + c.s64 + carry_bit.pred;
      else if (pI->is_hi())
        d.s32 = (t.s64 >> 32) + c.s32 + carry_bit.pred;
      else if (pI->is_lo())
        d.s32 = t.s32 + c.s32 + carry_bit.pred;
      else
        assert(0);
      break;
    case S64_TYPE:
      t.s64 = a.s64 * b.s64;
      assert(!pI->is_wide());
      assert(!pI->is_hi());
      assert(use_carry == false);
      if (pI->is_lo())
        d.s64 = t.s64 + c.s64 + carry_bit.pred;
      else
        assert(0);
      break;
    case U16_TYPE:
      t.u32 = a.u16 * b.u16;
      if (pI->is_wide())
        d.u32 = t.u32 + c.u32 + carry_bit.pred;
      else if (pI->is_hi())
        d.u16 = (t.u32 + c.u16 + carry_bit.pred) >> 16;
      else if (pI->is_lo())
        d.u16 = t.u16 + c.u16 + carry_bit.pred;
      else
        assert(0);
      carry = ((long long int)((long long int)t.u32 + c.u32 + carry_bit.pred) &
               0x100000000) >>
              32;
      break;
    case U32_TYPE:
      t.u64 = a.u32 * b.u32;
      if (pI->is_wide())
        d.u64 = t.u64 + c.u64 + carry_bit.pred;
      else if (pI->is_hi())
        d.u32 = (t.u64 + c.u32 + carry_bit.pred) >> 32;
      else if (pI->is_lo())
        d.u32 = t.u32 + c.u32 + carry_bit.pred;
      else
        assert(0);
      break;
    case U64_TYPE:
      t.u64 = a.u64 * b.u64;
      assert(!pI->is_wide());
      assert(!pI->is_hi());
      assert(use_carry == false);
      if (pI->is_lo())
        d.u64 = t.u64 + c.u64 + carry_bit.pred;
      else
        assert(0);
      break;
    case F16_TYPE: {
      // assert(0);
      // break;
      assert(use_carry == false);
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }
      d.f16 = a.f16 * b.f16 + c.f16;
      if (pI->saturation_mode()) {
        if (d.f16 < 0)
          d.f16 = 0;
        else if (d.f16 > 1.0f)
          d.f16 = 1.0f;
      }
      fesetround(orig_rm);
      break;
    }
    case F32_TYPE: {
      assert(use_carry == false);
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }
      d.f32 = a.f32 * b.f32 + c.f32;
      if (pI->saturation_mode()) {
        if (d.f32 < 0)
          d.f32 = 0;
        else if (d.f32 > 1.0f)
          d.f32 = 1.0f;
      }
      fesetround(orig_rm);
      break;
    }
    case F64_TYPE:
    case FF64_TYPE: {
      assert(use_carry == false);
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }
      d.f64 = a.f64 * b.f64 + c.f64;
      if (pI->saturation_mode()) {
        if (d.f64 < 0)
          d.f64 = 0;
        else if (d.f64 > 1.0f)
          d.f64 = 1.0;
      }
      fesetround(orig_rm);
      break;
    }
    default:
      assert(0);
      break;
  }
  thread->set_operand_value(dst, d, i_type, thread, pI, overflow, carry);
}

bool isNaN(float x) { return std::isnan(x); }

bool isNaN(double x) { return std::isnan(x); }

void max_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
      d.u16 = MY_MAX_I(a.u16, b.u16);
      break;
    case U32_TYPE:
      d.u32 = MY_MAX_I(a.u32, b.u32);
      break;
    case U64_TYPE:
      d.u64 = MY_MAX_I(a.u64, b.u64);
      break;
    case S16_TYPE:
      d.s16 = MY_MAX_I(a.s16, b.s16);
      break;
    case S32_TYPE:
      d.s32 = MY_MAX_I(a.s32, b.s32);
      break;
    case S64_TYPE:
      d.s64 = MY_MAX_I(a.s64, b.s64);
      break;
    case F32_TYPE:
      d.f32 = MY_MAX_F(a.f32, b.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = MY_MAX_F(a.f64, b.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void membar_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // handled by timing simulator
}

void min_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
      d.u16 = MY_MIN_I(a.u16, b.u16);
      break;
    case U32_TYPE:
      d.u32 = MY_MIN_I(a.u32, b.u32);
      break;
    case U64_TYPE:
      d.u64 = MY_MIN_I(a.u64, b.u64);
      break;
    case S16_TYPE:
      d.s16 = MY_MIN_I(a.s16, b.s16);
      break;
    case S32_TYPE:
      d.s32 = MY_MIN_I(a.s32, b.s32);
      break;
    case S64_TYPE:
      d.s64 = MY_MIN_I(a.s64, b.s64);
      break;
    case F32_TYPE:
      d.f32 = MY_MIN_F(a.f32, b.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = MY_MIN_F(a.f64, b.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

static int _count = 0;
void mov_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  //   if(thread->get_ctaid().x == 0 && thread->get_ctaid().y == 0 && thread->get_ctaid().z == 0)
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }
  // if(thread->get_tid().x == 6)
  //   if(pI->dst().name() == "%ssa_233")
  //     if(pI->src1().name() == "%ssa_166")
  //     printf("this is wrong\n");
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  unsigned i_type = pI->get_type();
  assert(src1.is_param_local() == 0);
  ptx_reg_t symbolicReg = thread->get_reg(dst.get_symbol());

  if ((src1.is_vector() || dst.is_vector()) && (i_type != BB64_TYPE) &&
      (i_type != BB128_TYPE) && (i_type != FF64_TYPE)) {
    // pack or unpack operation
    unsigned nbits_to_move;
    ptx_reg_t tmp_bits;

    switch (pI->get_type()) {
      case B16_TYPE:
        nbits_to_move = 16;
        break;
      case B32_TYPE:
      case U32_TYPE:
      case F32_TYPE:
        nbits_to_move = 32;
        break;
      case B64_TYPE:
        nbits_to_move = 64;
        break;
      default:
        printf(
            "Execution error: mov pack/unpack with unsupported type "
            "qualifier\n");
        assert(0);
        break;
    }

    if (src1.is_vector()) {
      unsigned nelem = src1.get_vect_nelem();
      ptx_reg_t v[4];
      thread->get_vector_operand_values(src1, v, nelem);

      unsigned bits_per_src_elem = nbits_to_move / nelem;
      for (unsigned i = 0; i < nelem; i++) {
        switch (bits_per_src_elem) {
          case 8:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u8) << (8 * i));
            break;
          case 16:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u16) << (16 * i));
            break;
          case 32:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u32) << (32 * i));
            break;
          default:
            printf(
                "Execution error: mov pack/unpack with unsupported source/dst "
                "size ratio (src)\n");
            assert(0);
            break;
        }
      }
    } else {
      data = thread->get_operand_value(src1, dst, i_type, thread, 1);

      switch (pI->get_type()) {
        case B16_TYPE:
          tmp_bits.u16 = data.u16;
          break;
        case B32_TYPE:
        case U32_TYPE:
          tmp_bits.u32 = data.u32;
          break;
        case F32_TYPE:
          tmp_bits.f32 = data.f32;
          break;
        case B64_TYPE:
          tmp_bits.u64 = data.u64;
          break;
        default:
          assert(0);
          break;
      }
    }

    if (dst.is_vector()) {
      unsigned nelem = dst.get_vect_nelem();
      ptx_reg_t v[4];
      unsigned bits_per_dst_elem = nbits_to_move / nelem;
      for (unsigned i = 0; i < nelem; i++) {
        switch (bits_per_dst_elem) {
          case 8:
            v[i].u8 = (tmp_bits.u64 >> (8 * i)) & ((unsigned long long)0xFF);
            break;
          case 16:
            v[i].u16 =
                (tmp_bits.u64 >> (16 * i)) & ((unsigned long long)0xFFFF);
            break;
          case 32:
            v[i].u32 =
                (tmp_bits.u64 >> (32 * i)) & ((unsigned long long)0xFFFFFFFF);
            break;
          default:
            printf(
                "Execution error: mov pack/unpack with unsupported source/dst "
                "size ratio (dst)\n");
            assert(0);
            break;
        }
      }
      thread->set_vector_operand_values(dst, v[0], v[1], v[2], v[3]);
    } else {
      thread->set_operand_value(dst, tmp_bits, i_type, thread, pI);
    }
  } else if (i_type == PRED_TYPE and src1.is_literal() == true) {
    // in ptx, literal input translate to predicate as 0 = false and 1 = true
    // we have adopted the opposite to simplify implementation of zero flags in
    // ptxplus
    data = thread->get_operand_value(src1, dst, i_type, thread, 1);

    ptx_reg_t finaldata;
    finaldata.pred = (data.u32 == 0) ? 1 : 0;  // setting zero-flag in predicate
    thread->set_operand_value(dst, finaldata, i_type, thread, pI);
  } else {
    data = thread->get_operand_value(src1, dst, i_type, thread, 1);

    thread->set_operand_value(dst, data, i_type, thread, pI);
  }
}

void mul24_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // src1_data = srcOperandModifiers(src1_data, src1, dst, i_type, thread);
  // src2_data = srcOperandModifiers(src2_data, src2, dst, i_type, thread);

  src1_data.mask_and(0, 0x00FFFFFF);
  src2_data.mask_and(0, 0x00FFFFFF);

  switch (i_type) {
    case S32_TYPE:
      if (src1_data.get_bit(23)) src1_data.mask_or(0xFFFFFFFF, 0xFF000000);
      if (src2_data.get_bit(23)) src2_data.mask_or(0xFFFFFFFF, 0xFF000000);
      data.s64 = src1_data.s64 * src2_data.s64;
      break;
    case U32_TYPE:
      data.u64 = src1_data.u64 * src2_data.u64;
      break;
    default:
      printf(
          "GPGPU-Sim PTX: Execution error - type mismatch with instruction\n");
      assert(0);
      break;
  }

  if (pI->is_hi()) {
    data.u64 = data.u64 >> 16;
    data.mask_and(0, 0xFFFFFFFF);
  } else if (pI->is_lo()) {
    data.mask_and(0, 0xFFFFFFFF);
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void mul_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  //   if(thread->get_ctaid().x == 0 && thread->get_ctaid().y == 0 && thread->get_ctaid().z == 0)
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  ptx_reg_t d, t;

  unsigned i_type = pI->get_type();
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  unsigned rounding_mode = pI->rounding_mode();

  switch (i_type) {
    case S16_TYPE:
      t.s32 = ((int)a.s16) * ((int)b.s16);
      if (pI->is_wide())
        d.s32 = t.s32;
      else if (pI->is_hi())
        d.s16 = (t.s32 >> 16);
      else if (pI->is_lo())
        d.s16 = t.s16;
      else
        assert(0);
      break;
    case S32_TYPE:
      t.s64 = ((long long)a.s32) * ((long long)b.s32);
      if (pI->is_wide())
        d.s64 = t.s64;
      else if (pI->is_hi())
        d.s32 = (t.s64 >> 32);
      else if (pI->is_lo())
        d.s32 = t.s32;
      else
        assert(0);
      break;
    case S64_TYPE:
      t.s64 = a.s64 * b.s64;
      assert(!pI->is_wide());
      assert(!pI->is_hi());
      if (pI->is_lo())
        d.s64 = t.s64;
      else
        assert(0);
      break;
    case U16_TYPE:
      t.u32 = ((unsigned)a.u16) * ((unsigned)b.u16);
      if (pI->is_wide())
        d.u32 = t.u32;
      else if (pI->is_lo())
        d.u16 = t.u16;
      else if (pI->is_hi())
        d.u16 = (t.u32 >> 16);
      else
        assert(0);
      break;
    case U32_TYPE:
      t.u64 = ((unsigned long long)a.u32) * ((unsigned long long)b.u32);
      if (pI->is_wide())
        d.u64 = t.u64;
      else if (pI->is_lo())
        d.u32 = t.u32;
      else if (pI->is_hi())
        d.u32 = (t.u64 >> 32);
      else
        assert(0);
      break;
    case U64_TYPE:
      t.u64 = a.u64 * b.u64;
      assert(!pI->is_wide());
      assert(!pI->is_hi());
      if (pI->is_lo())
        d.u64 = t.u64;
      else
        assert(0);
      break;
    case F16_TYPE: {
      // assert(0);
      // break;
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }

      d.f16 = a.f16 * b.f16;

      if (pI->saturation_mode()) {
        if (d.f16 < 0)
          d.f16 = 0;
        else if (d.f16 > 1.0f)
          d.f16 = 1.0f;
      }
      fesetround(orig_rm);
      break;
    }
    case F32_TYPE: {
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }

      d.f32 = a.f32 * b.f32;

      if (pI->saturation_mode()) {
        if (d.f32 < 0)
          d.f32 = 0;
        else if (d.f32 > 1.0f)
          d.f32 = 1.0f;
      }
      fesetround(orig_rm);
      break;
    }
    case F64_TYPE:
    case FF64_TYPE: {
      int orig_rm = fegetround();
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);
          break;
        default:
          assert(0);
          break;
      }
      d.f64 = a.f64 * b.f64;
      if (pI->saturation_mode()) {
        if (d.f64 < 0)
          d.f64 = 0;
        else if (d.f64 > 1.0f)
          d.f64 = 1.0;
      }
      fesetround(orig_rm);
      break;
    }
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void neg_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned to_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, to_type, thread, 1);

  switch (to_type) {
    case S8_TYPE:
    case S16_TYPE:
    case S32_TYPE:
    case S64_TYPE:
      data.s64 = 0 - src1_data.s64;
      break;  // seems buggy, but not (just ignore higher bits)
    case U8_TYPE:
    case U16_TYPE:
    case U32_TYPE:
    case U64_TYPE:
      assert(0);
      break;
    case F16_TYPE:
      data.f16 = 0.0f - src1_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = 0.0f - src1_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = 0.0f - src1_data.f64;
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, to_type, thread, pI);
}

// nandn bitwise negates second operand then bitwise nands with the first
// operand
void nandn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = (~src1_data.pred & src2_data.pred);
  else
    data.u64 = ~(src1_data.u64 & ~src2_data.u64);

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

// norn bitwise negates first operand then bitwise ands with the second operand
void norn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(src1_data.pred & ~(src2_data.pred));
  else
    data.u64 = ~(src1_data.u64) & src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void not_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case PRED_TYPE:
      d.pred = (~(a.pred) & 0x000F);
      break;
    case B16_TYPE:
      d.u16 = ~a.u16;
      break;
    case B32_TYPE:
      d.u32 = ~a.u32;
      break;
    case B64_TYPE:
      d.u64 = ~a.u64;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void or_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(~(src1_data.pred) | ~(src2_data.pred));
  else
    data.u64 = src1_data.u64 | src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void orn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(~(src1_data.pred) | (src2_data.pred));
  else
    data.u64 = src1_data.u64 | ~src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void pmevent_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void popc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();

  unsigned i_type = pI->get_type();
  src_data = thread->get_operand_value(src, dst, i_type, thread, 1);

  switch (i_type) {
    case B32_TYPE: {
      std::bitset<32> mask(src_data.u32);
      data.u32 = mask.count();
    } break;
    case B64_TYPE: {
      std::bitset<64> mask(src_data.u64);
      data.u32 = mask.count();
    } break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}
void prefetch_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void prefetchu_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

int prmt_mode_present(int mode) {
  int returnval = 0;
  switch (mode) {
    case PRMT_F4E_MODE:
    case PRMT_B4E_MODE:
    case PRMT_RC8_MODE:
    case PRMT_RC16_MODE:
    case PRMT_ECL_MODE:
    case PRMT_ECR_MODE:
      returnval = 1;
      break;
    default:
      break;
  }
  return returnval;
}
int read_byte(int mode, int control, int d_sel_index, signed long long value) {
  int returnval = 0;
  int prmt_f4e_mode[4][4] = {
      {0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}};
  int prmt_b4e_mode[4][4] = {
      {0, 7, 6, 5}, {1, 0, 7, 6}, {2, 1, 0, 7}, {3, 2, 1, 0}};
  int prmt_rc8_mode[4][4] = {
      {0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}};
  int prmt_ecl_mode[4][4] = {
      {0, 1, 2, 3}, {1, 1, 2, 3}, {2, 2, 2, 3}, {3, 3, 3, 3}};
  int prmt_ecr_mode[4][4] = {
      {0, 0, 0, 0}, {0, 1, 1, 1}, {0, 1, 2, 2}, {0, 1, 2, 3}};
  int prmt_rc16_mode[4][4] = {
      {0, 1, 0, 1}, {2, 3, 2, 3}, {0, 1, 0, 1}, {2, 3, 2, 3}};

  if (!prmt_mode_present(mode)) {
    if (control & 0x8) {
      returnval = 0xff;
    } else {
      returnval = (value >> (8 * control)) & 0xff;
    }
  } else {
    switch (mode) {
      case PRMT_F4E_MODE:
        returnval = prmt_f4e_mode[control][d_sel_index];
        break;
      case PRMT_B4E_MODE:
        returnval = prmt_b4e_mode[control][d_sel_index];
        break;
      case PRMT_RC8_MODE:
        returnval = prmt_rc8_mode[control][d_sel_index];
        break;
      case PRMT_ECL_MODE:
        returnval = prmt_ecl_mode[control][d_sel_index];
        break;
      case PRMT_ECR_MODE:
        returnval = prmt_ecr_mode[control][d_sel_index];
        break;
      case PRMT_RC16_MODE:
        returnval = prmt_rc16_mode[control][d_sel_index];
        break;
        // Change the default from printing "ERROR" to just asserting
      default:
        assert(false);
    }
  }
  return (returnval << 8 * d_sel_index);
}

void prmt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, src3_data, tmpdata, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  unsigned mode = pI->prmt_op();
  unsigned i_type = pI->get_type();

  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);

  tmpdata.s64 = src1_data.s32 | (src2_data.s64 << 32);
  int ctl[4];

  if (!prmt_mode_present(mode)) {
    ctl[0] = (src3_data.s32 >> 0) & 0xf;
    ctl[1] = (src3_data.s32 >> 4) & 0xf;
    ctl[2] = (src3_data.s32 >> 8) & 0xf;
    ctl[3] = (src3_data.s32 >> 12) & 0xf;
  } else {
    ctl[0] = ctl[1] = ctl[2] = ctl[3] = (src3_data.s32 >> 0) & 0x3;
  }

  data.s32 = 0;
  data.s32 = data.s32 | read_byte(mode, ctl[0], 0, tmpdata.s64);  // First
                                                                  // byte-0
  data.s32 =
      data.s32 | read_byte(mode, ctl[1], 1, tmpdata.s64);  // Second byte-1
  data.s32 = data.s32 | read_byte(mode, ctl[2], 2, tmpdata.s64);  // Third
                                                                  // byte-2
  data.s32 =
      data.s32 | read_byte(mode, ctl[3], 3, tmpdata.s64);  // Fourth byte-3

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void rcp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      data.f32 = 1.0f / src1_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = 1.0f / src1_data.f64;
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void red_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void rem_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case S32_TYPE:
      data.s32 = src1_data.s32 % src2_data.s32;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 % src2_data.s64;
      break;
    case U32_TYPE:
      data.u32 = src1_data.u32 % src2_data.u32;
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 % src2_data.u64;
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void ret_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
    // if(thread->get_ctaid().x == 2 && thread->get_ctaid().y == 89 && thread->get_ctaid().z == 0)
    VSIM_DPRINTF("gpgpusim: return from function in %s\n", pI->source_file());
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }

  bool empty = thread->callstack_pop();
  if (empty) {
    thread->set_done();
    thread->exitCore();
    thread->registerExit();
  }
}

// Ptxplus version of ret instruction.
void retp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  bool empty = thread->callstack_pop_plus();
  if (empty) {
    thread->set_done();
    thread->exitCore();
    thread->registerExit();
  }
}

void rsqrt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      if (a.f32 < 0) {
        d.u64 = 0;
        d.u64 = 0x7fc00000;  // NaN
      } else if (a.f32 == 0) {
        d.u64 = 0;
        d.u32 = 0x7f800000;  // Inf
      } else
        d.f32 = cuda_math::__internal_accurate_fdividef(1.0f, sqrtf(a.f32));
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (a.f32 < 0) {
        d.u64 = 0;
        d.u32 = 0x7fc00000;  // NaN
        float x = d.f32;
        d.f64 = (double)x;
      } else if (a.f32 == 0) {
        d.u64 = 0;
        d.u32 = 0x7f800000;  // Inf
        float x = d.f32;
        d.f64 = (double)x;
      } else
        d.f64 = 1.0 / sqrt(a.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

#define SAD(d, a, b, c) d = c + ((a < b) ? (b - a) : (a - b))

void sad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, c, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
      SAD(d.u16, a.u16, b.u16, c.u16);
      break;
    case U32_TYPE:
      SAD(d.u32, a.u32, b.u32, c.u32);
      break;
    case U64_TYPE:
      SAD(d.u64, a.u64, b.u64, c.u64);
      break;
    case S16_TYPE:
      SAD(d.s16, a.s16, b.s16, c.s16);
      break;
    case S32_TYPE:
      SAD(d.s32, a.s32, b.s32, c.s32);
      break;
    case S64_TYPE:
      SAD(d.s64, a.s64, b.s64, c.s64);
      break;
    case F32_TYPE:
      SAD(d.f32, a.f32, b.f32, c.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      SAD(d.f64, a.f64, b.f64, c.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void selp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  ptx_reg_t a, b, c, d;

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  // predicate value was changed so the lowest bit being set means the zero flag
  // is set. As a result, the value of c.pred must be inverted to get proper
  // behavior
  d = (!(c.pred & 0x0001)) ? a : b;

  thread->set_operand_value(dst, d, PRED_TYPE, thread, pI);
}

bool isFloat(int type) {
  switch (type) {
    case F16_TYPE:
    case F32_TYPE:
    case F64_TYPE:
    case FF64_TYPE:
      return true;
    default:
      return false;
  }
}

bool CmpOp(int type, ptx_reg_t a, ptx_reg_t b, unsigned cmpop) {
  bool t = false;

  // TODO-LUCY: Find out what "td" is used for
  bool td = false;

  switch (type) {
    case B16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u16 == b.u16);
          break;
        case NE_OPTION:
          t = (a.u16 != b.u16);
          break;
        default:
          assert(0);
      }

    case B32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u32 == b.u32);
          break;
        case NE_OPTION:
          t = (a.u32 != b.u32);
          break;
        default:
          assert(0);
      }
    case B64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u64 == b.u64);
          break;
        case NE_OPTION:
          t = (a.u64 != b.u64);
          break;
        default:
          assert(0);
      }
      break;
    case S8_TYPE:
    case S16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s16 == b.s16);
          break;
        case NE_OPTION:
          t = (a.s16 != b.s16);
          break;
        case LT_OPTION:
          t = (a.s16 < b.s16);
          break;
        case LE_OPTION:
          t = (a.s16 <= b.s16);
          break;
        case GT_OPTION:
          t = (a.s16 > b.s16);
          break;
        case GE_OPTION:
          t = (a.s16 >= b.s16);
          break;
        default:
          assert(0);
      }
      break;
    case S32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s32 == b.s32);
          break;
        case NE_OPTION:
          t = (a.s32 != b.s32);
          break;
        case LT_OPTION:
          t = (a.s32 < b.s32);
          break;
        case LE_OPTION:
          t = (a.s32 <= b.s32);
          break;
        case GT_OPTION:
          t = (a.s32 > b.s32);
          break;
        case GE_OPTION:
          t = (a.s32 >= b.s32);
          break;
        default:
          assert(0);
      }
      break;
    case S64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s64 == b.s64);
          break;
        case NE_OPTION:
          t = (a.s64 != b.s64);
          break;
        case LT_OPTION:
          t = (a.s64 < b.s64);
          break;
        case LE_OPTION:
          t = (a.s64 <= b.s64);
          break;
        case GT_OPTION:
          t = (a.s64 > b.s64);
          break;
        case GE_OPTION:
          t = (a.s64 >= b.s64);
          break;
        default:
          assert(0);
      }
      break;
    case U8_TYPE:
    case U16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u16 == b.u16);
          break;
        case NE_OPTION:
          t = (a.u16 != b.u16);
          break;
        case LT_OPTION:
          t = (a.u16 < b.u16);
          break;
        case LE_OPTION:
          t = (a.u16 <= b.u16);
          break;
        case GT_OPTION:
          t = (a.u16 > b.u16);
          break;
        case GE_OPTION:
          t = (a.u16 >= b.u16);
          break;
        case LO_OPTION:
          t = (a.u16 < b.u16);
          break;
        case LS_OPTION:
          t = (a.u16 <= b.u16);
          break;
        case HI_OPTION:
          t = (a.u16 > b.u16);
          break;
        case HS_OPTION:
          t = (a.u16 >= b.u16);
          break;
        default:
          assert(0);
      }
      break;
    case U32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u32 == b.u32);
          break;
        case NE_OPTION:
          t = (a.u32 != b.u32);
          break;
        case LT_OPTION:
          t = (a.u32 < b.u32);
          break;
        case LE_OPTION:
          t = (a.u32 <= b.u32);
          break;
        case GT_OPTION:
          t = (a.u32 > b.u32);
          break;
        case GE_OPTION:
          t = (a.u32 >= b.u32);
          break;
        case LO_OPTION:
          t = (a.u32 < b.u32);
          break;
        case LS_OPTION:
          t = (a.u32 <= b.u32);
          break;
        case HI_OPTION:
          t = (a.u32 > b.u32);
          break;
        case HS_OPTION:
          t = (a.u32 >= b.u32);
          break;
        default:
          assert(0);
      }
      break;
    case U64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u64 == b.u64);
          break;
        case NE_OPTION:
          t = (a.u64 != b.u64);
          break;
        case LT_OPTION:
          t = (a.u64 < b.u64);
          break;
        case LE_OPTION:
          t = (a.u64 <= b.u64);
          break;
        case GT_OPTION:
          t = (a.u64 > b.u64);
          break;
        case GE_OPTION:
          t = (a.u64 >= b.u64);
          break;
        case LO_OPTION:
          t = (a.u64 < b.u64);
          break;
        case LS_OPTION:
          t = (a.u64 <= b.u64);
          break;
        case HI_OPTION:
          t = (a.u64 > b.u64);
          break;
        case HS_OPTION:
          t = (a.u64 >= b.u64);
          break;
        default:
          assert(0);
      }
      break;
    case F16_TYPE:
      assert(0);
      break;
    case F32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.f32 == b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case NE_OPTION:
          t = (a.f32 != b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case LT_OPTION:
          t = (a.f32 < b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case LE_OPTION:
          t = (a.f32 <= b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case GT_OPTION:
          t = (a.f32 > b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case GE_OPTION:
          t = (a.f32 >= b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case EQU_OPTION:
          t = (a.f32 == b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case NEU_OPTION:
          t = (a.f32 != b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case LTU_OPTION:
          t = (a.f32 < b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case LEU_OPTION:
          t = (a.f32 <= b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case GTU_OPTION:
          t = (a.f32 > b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case GEU_OPTION:
          t = (a.f32 >= b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case NUM_OPTION:
          t = !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case NAN_OPTION:
          t = isNaN(a.f32) || isNaN(b.f32);
          break;
        default:
          assert(0);
      }
      break;
    case F64_TYPE:
    case FF64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.f64 == b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case NE_OPTION:
          t = (a.f64 != b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case LT_OPTION:
          t = (a.f64 < b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case LE_OPTION:
          t = (a.f64 <= b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case GT_OPTION:
          t = (a.f64 > b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case GE_OPTION:
          t = (a.f64 >= b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case EQU_OPTION:
          t = (a.f64 == b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case NEU_OPTION:
          t = (a.f64 != b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case LTU_OPTION:
          t = (a.f64 < b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case LEU_OPTION:
          t = (a.f64 <= b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case GTU_OPTION:
          t = (a.f64 > b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case GEU_OPTION:
          t = (a.f64 >= b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case NUM_OPTION:
          t = !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case NAN_OPTION:
          t = isNaN(a.f64) || isNaN(b.f64);
          break;
        default:
          assert(0);
      }
      break;
    default:
      assert(0);
      break;
  }

  return t;
}

void setp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b;

  int t = 0;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  assert(pI->get_num_operands() <
         4);  // or need to deal with "c" operand / boolOp

  unsigned type = pI->get_type();
  unsigned cmpop = pI->get_cmpop();
  a = thread->get_operand_value(src1, dst, type, thread, 1);
  b = thread->get_operand_value(src2, dst, type, thread, 1);

  t = CmpOp(type, a, b, cmpop);

  ptx_reg_t data;

  // the way ptxplus handles the zero flag, 1 = false and 0 = true
  data.pred =
      (t ==
       0);  // inverting predicate since ptxplus uses "1" for a set zero flag

  thread->set_operand_value(dst, data, PRED_TYPE, thread, pI);
}

void set_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b;

  int t = 0;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  assert(pI->get_num_operands() <
         4);  // or need to deal with "c" operand / boolOp

  unsigned src_type = pI->get_type2();
  unsigned cmpop = pI->get_cmpop();

  a = thread->get_operand_value(src1, dst, src_type, thread, 1);
  b = thread->get_operand_value(src2, dst, src_type, thread, 1);

  // Take abs of first operand if needed
  if (pI->is_abs()) {
    switch (src_type) {
      case S16_TYPE:
        a.s16 = my_abs(a.s16);
        break;
      case S32_TYPE:
        a.s32 = my_abs(a.s32);
        break;
      case S64_TYPE:
        a.s64 = my_abs(a.s64);
        break;
      case U16_TYPE:
        a.u16 = a.u16;
        break;
      case U32_TYPE:
        a.u32 = my_abs(a.u32);
        break;
      case U64_TYPE:
        a.u64 = my_abs(a.u64);
        break;
      case F32_TYPE:
        a.f32 = my_abs(a.f32);
        break;
      case F64_TYPE:
      case FF64_TYPE:
        a.f64 = my_abs(a.f64);
        break;
      default:
        printf("Execution error: type mismatch with instruction\n");
        assert(0);
        break;
    }
  }

  t = CmpOp(src_type, a, b, cmpop);

  ptx_reg_t data;
  if (isFloat(pI->get_type())) {
    data.f32 = (t != 0) ? 1.0f : 0.0f;
  } else {
    data.u32 = (t != 0) ? 0xFFFFFFFF : 0;
  }

  thread->set_operand_value(dst, data, pI->get_type(), thread, pI);
}

void shfl_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst) {
  unsigned i_type = pI->get_type();
  int tid;

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size();
  else
    tid = inst.warp_id() * core->get_warp_size();

  ptx_thread_info *thread = core->get_thread_info()[tid];
  ptx_warp_info *warp_info = thread->m_warp_info;
  int lane = warp_info->get_done_threads();
  thread = core->get_thread_info()[tid + lane];

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  int bval = (thread->get_operand_value(src2, dst, i_type, thread, 1)).u32;
  int cval = (thread->get_operand_value(src3, dst, i_type, thread, 1)).u32;
  int mask = cval >> 8;
  bval &= 0x1F;
  cval &= 0x1F;

  int maxLane = (lane & mask) | (cval & ~mask);
  int minLane = lane & mask;

  int src_idx;
  unsigned p;
  switch (pI->shfl_op()) {
    case UP_OPTION:
      src_idx = lane - bval;
      p = (src_idx >= maxLane);
      break;
    case DOWN_OPTION:
      src_idx = lane + bval;
      p = (src_idx <= maxLane);
      break;
    case BFLY_OPTION:
      src_idx = lane ^ bval;
      p = (src_idx <= maxLane);
      break;
    case IDX_OPTION:
      src_idx = minLane | (bval & ~mask);
      p = (src_idx <= maxLane);
      break;
    default:
      printf("GPGPU-Sim PTX: ERROR: Invalid shfl option\n");
      assert(0);
      break;
  }
  // copy from own lane
  if (!p) src_idx = lane;

  // copy input from lane src_idx
  ptx_reg_t data;
  if (inst.active(src_idx)) {
    ptx_thread_info *source = core->get_thread_info()[tid + src_idx];
    data = source->get_operand_value(src1, dst, i_type, source, 1);
  } else {
    printf(
        "GPGPU-Sim PTX: WARNING: shfl input value unpredictable for inactive "
        "threads in a warp\n");
    data.u32 = 0;
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);

  /*
  TODO: deal with predicates appropriately using the following pseudocode:
  if (!isGuardPredicateTrue(src_idx)) {
          printf("GPGPU-Sim PTX: WARNING: shfl input value unpredictable for
  predicated-off threads in a warp\n");
  }
  if (dest predicate selected) data.pred = p;
  */

  // keep track of the number of threads that have executed in the warp
  warp_info->inc_done_threads();
  if (warp_info->get_done_threads() == inst.active_count()) {
    warp_info->reset_done_threads();
  }
}

void shl_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case B16_TYPE:
    case U16_TYPE:
      if (b.u16 >= 16)
        d.u16 = 0;
      else
        d.u16 = (unsigned short)((a.u16 << b.u16) & 0xFFFF);
      break;
    case B32_TYPE:
    case U32_TYPE:
      if (b.u32 >= 32)
        d.u32 = 0;
      else
        d.u32 = (unsigned)((a.u32 << b.u32) & 0xFFFFFFFF);
      break;
    case B64_TYPE:
    case U64_TYPE:
      if (b.u32 >= 64)
        d.u64 = 0;
      else
        d.u64 = (a.u64 << b.u64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void shr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
    case B16_TYPE:
      if (b.u16 < 16)
        d.u16 = (unsigned short)((a.u16 >> b.u16) & 0xFFFF);
      else
        d.u16 = 0;
      break;
    case U32_TYPE:
    case B32_TYPE:
      if (b.u32 < 32)
        d.u32 = (unsigned)((a.u32 >> b.u32) & 0xFFFFFFFF);
      else
        d.u32 = 0;
      break;
    case U64_TYPE:
    case B64_TYPE:
      if (b.u32 < 64)
        d.u64 = (a.u64 >> b.u64);
      else
        d.u64 = 0;
      break;
    case S16_TYPE:
      if (b.u16 < 16)
        d.s64 = (a.s16 >> b.s16);
      else {
        if (a.s16 < 0) {
          d.s64 = -1;
        } else {
          d.s64 = 0;
        }
      }
      break;
    case S32_TYPE:
      if (b.u32 < 32)
        d.s64 = (a.s32 >> b.s32);
      else {
        if (a.s32 < 0) {
          d.s64 = -1;
        } else {
          d.s64 = 0;
        }
      }
      break;
    case S64_TYPE:
      if (b.u64 < 64)
        d.s64 = (a.s64 >> b.u64);
      else {
        if (a.s64 < 0) {
          if (b.s32 < 0) {
            d.u64 = -1;
            d.s32 = 0;
          } else {
            d.s64 = -1;
          }
        } else {
          d.s64 = 0;
        }
      }
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void sin_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      d.f32 = sin(a.f32);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void slct_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  ptx_reg_t a, b, c, d;

  unsigned i_type = pI->get_type();
  unsigned c_type = pI->get_type2();
  bool t = false;
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, c_type, thread, 1);

  switch (c_type) {
    case S32_TYPE:
      t = c.s32 >= 0;
      break;
    case F32_TYPE:
      t = c.f32 >= 0;
      break;
    default:
      assert(0);
  }

  switch (i_type) {
    case B16_TYPE:
    case S16_TYPE:
    case U16_TYPE:
      d.u16 = t ? a.u16 : b.u16;
      break;
    case F32_TYPE:
    case B32_TYPE:
    case S32_TYPE:
    case U32_TYPE:
      d.u32 = t ? a.u32 : b.u32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
    case B64_TYPE:
    case S64_TYPE:
    case U64_TYPE:
      d.u64 = t ? a.u64 : b.u64;
      break;
    default:
      assert(0);
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void sqrt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      if (a.f32 < 0)
        d.f32 = nanf("");
      else
        d.f32 = sqrt(a.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (a.f64 < 0)
        d.f64 = nan("");
      else
        d.f64 = sqrt(a.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

void sst_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_instruction *cpI = const_cast<ptx_instruction *>(pI);  // constant
  const operand_info &dst = cpI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  unsigned type = pI->get_type();
  ptx_reg_t dst_data = thread->get_operand_value(dst, dst, type, thread, 1);
  ptx_reg_t src1_data = thread->get_operand_value(src1, src1, type, thread, 1);
  ptx_reg_t src2_data = thread->get_operand_value(src2, src1, type, thread, 1);
  ptx_reg_t src3_data = thread->get_operand_value(src3, src1, type, thread, 1);
  memory_space_t space = pI->get_space();
  memory_space *mem = NULL;
  addr_t addr =
      src2_data.u32 * 4;  // this assumes sstarr memory starts at address 0
  assert(0); //32 bit address
  ptx_cta_info *cta_info = thread->m_cta_info;

  decode_space(space, thread, src1, mem, addr);

  size_t size;
  int t;
  type_info_key::type_decode(type, size, t);

  // store data in sstarr memory
  mem->write(addr, size / 8, &src3_data.s64, thread, pI);

  // sync threads
  cpI->set_bar_id(16);  // use 16 for sst because bar uses an int from 0-15

  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = space;
  thread->m_last_dram_callback.function = bar_callback;
  thread->m_last_dram_callback.instruction = cpI;

  // the last thread that executes loads all of the data back from sstarr memory
  int NUM_THREADS = cta_info->num_threads();
  cta_info->inc_bar_threads();
  if (NUM_THREADS == cta_info->get_bar_threads()) {
    unsigned offset = 0;
    addr = 0;
    ptx_reg_t data;
    float sstarr_fdata[NUM_THREADS];
    signed long long sstarr_ldata[NUM_THREADS];
    // loop through all of the threads
    for (int tid = 0; tid < NUM_THREADS; tid++) {
      data.u64 = 0;
      mem->read(addr + (tid * 4), size / 8, &data.s64);
      sstarr_fdata[tid] = data.f32;
      sstarr_ldata[tid] = data.s64;
    }

    // squeeze the zeros out of the array and store data back into original
    // array
    mem = NULL;
    addr = src1_data.u32;
    space.set_type(global_space);
    decode_space(space, thread, src1, mem, addr);
    // store nonzero entries and indices
    for (int tid = 0; tid < NUM_THREADS; tid++) {
      if (sstarr_fdata[tid] != 0) {
        float ftid = (float)tid;
        mem->write(addr + (offset * 4), size / 8, &sstarr_ldata[tid], thread,
                   pI);
        mem->write(addr + ((NUM_THREADS + offset) * 4), size / 8, &ftid, thread,
                   pI);
        offset++;
      }
    }
    // store the number of nonzero elements in the array
    data = thread->get_operand_value(src1, dst, type, thread, 1);
    data.s64 += 4 * (offset - 1);
    thread->set_operand_value(dst, data, type, thread, pI);

    // fill the rest of the array with zeros (dst should always have a 0 in it)
    while (offset < NUM_THREADS) {
      mem->write(addr + (offset * 4), size / 8, &dst_data.s64, thread, pI);
      offset++;
    }

    cta_info->reset_bar_threads();
    thread->m_last_effective_address = addr + (NUM_THREADS - 1) * 4;
    thread->m_last_memory_space = space;
  }
}

void ssy_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // printf("Execution Warning: unimplemented ssy instruction is treated as a
  // nop\n");
  // TODO: add implementation
}

void st_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
    // if(thread->get_ctaid().x == 2 && thread->get_ctaid().y == 89 && thread->get_ctaid().z == 0)
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();  // may be scalar or vector of regs
  unsigned type = pI->get_type();
  ptx_reg_t addr_reg = thread->get_operand_value(dst, dst, type, thread, 1);
  ptx_reg_t data;
  memory_space_t space = pI->get_space();
  unsigned vector_spec = pI->get_vector();

  memory_space *mem = NULL;
  // addr_t addr = addr_reg.u32;
  addr_t addr = addr_reg.u64;

  decode_space(space, thread, dst, mem, addr);

  size_t size;
  int t;
  type_info_key::type_decode(type, size, t);

  if (!vector_spec) {
    data = thread->get_operand_value(src1, dst, type, thread, 1);
    mem->write(addr, size / 8, &data.s64, thread, pI);
    // assert(size == 32);
    // memcpy(address, &data.s64, size / 8);
    // *address = data.f32;
  } else {
    assert (0);
    if (vector_spec == V2_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[2];
      thread->get_vector_operand_values(src1, ptx_regs, 2);
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      delete[] ptx_regs;
    }
    if (vector_spec == V3_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[3];
      thread->get_vector_operand_values(src1, ptx_regs, 3);
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      mem->write(addr + 2 * size / 8, size / 8, &ptx_regs[2].s64, thread, pI);
      delete[] ptx_regs;
    }
    if (vector_spec == V4_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[4];
      thread->get_vector_operand_values(src1, ptx_regs, 4);
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      mem->write(addr + 2 * size / 8, size / 8, &ptx_regs[2].s64, thread, pI);
      mem->write(addr + 3 * size / 8, size / 8, &ptx_regs[3].s64, thread, pI);
      delete[] ptx_regs;
    }
  }
  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = space;
}

void sub_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;
  int overflow = 0;
  int carry = 0;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // performs addition. Sets carry and overflow if needed.
  // the constant is added in during subtraction so the carry bit is set
  // properly.
  switch (i_type) {
    case S8_TYPE:
      data.s64 = (src1_data.s64 & 0xFF) - (src2_data.s64 & 0xFF) + 0x100;
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) != 0) {
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;
      }
      carry = (data.s32 & 0x100) >> 8;
      break;
    case S16_TYPE:
      data.s64 = (src1_data.s64 & 0xFFFF) - (src2_data.s64 & 0xFFFF) + 0x10000;
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) != 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.s32 & 0x10000) >> 16;
      break;
    case S32_TYPE:
      data.s64 = (src1_data.s64 & 0xFFFFFFFF) - (src2_data.s64 & 0xFFFFFFFF) +
                 0x100000000;
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) != 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = ((data.u64) >> 32) & 0x0001;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 - src2_data.s64;
      break;
    case B8_TYPE:
    case U8_TYPE:
      data.u64 = (src1_data.u64 & 0xFF) - (src2_data.u64 & 0xFF) + 0x100;
      carry = (data.u64 & 0x100) >> 8;
      break;
    case B16_TYPE:
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) - (src2_data.u64 & 0xFFFF) + 0x10000;
      carry = (data.u64 & 0x10000) >> 16;
      break;
    case B32_TYPE:
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) - (src2_data.u64 & 0xFFFFFFFF) +
                 0x100000000;
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case B64_TYPE:
    case U64_TYPE:
      data.u64 = src1_data.u64 - src2_data.u64;
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 - src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 - src2_data.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 - src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);
}

void nop_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // Do nothing
}

void subc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void suld_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void sured_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void sust_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void suq_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

union intfloat {
  int a;
  float b;
};

float reduce_precision(float x, unsigned bits) {
  intfloat tmp;
  tmp.b = x;
  int v = tmp.a;
  int man = v & ((1 << 23) - 1);
  int mask = ((1 << bits) - 1) << (23 - bits);
  int nv = (v & ((-1) - ((1 << 23) - 1))) | (mask & man);
  tmp.a = nv;
  float result = tmp.b;
  return result;
}

unsigned wrap(unsigned x, unsigned y, unsigned mx, unsigned my,
              size_t elem_size) {
  unsigned nx = (mx + x) % mx;
  unsigned ny = (my + y) % my;
  return nx + mx * ny;
}

unsigned clamp(unsigned x, unsigned y, unsigned mx, unsigned my,
               size_t elem_size) {
  unsigned nx = x;
  while (nx >= mx) nx -= elem_size;
  unsigned ny = (y >= my) ? my - 1 : y;
  return nx + mx * ny;
}

typedef unsigned (*texAddr_t)(unsigned x, unsigned y, unsigned mx, unsigned my,
                              size_t elem_size);
float tex_linf_sampling(memory_space *mem, unsigned tex_array_base, int x,
                        int y, unsigned int width, unsigned int height,
                        size_t elem_size, float alpha, float beta,
                        texAddr_t b_lim) {
  float Tij;
  float Ti1j;
  float Tij1;
  float Ti1j1;

  mem->read(tex_array_base + b_lim(x, y, width, height, elem_size), 4, &Tij);
  mem->read(tex_array_base + b_lim(x + elem_size, y, width, height, elem_size),
            4, &Ti1j);
  mem->read(tex_array_base + b_lim(x, y + 1, width, height, elem_size), 4,
            &Tij1);
  mem->read(
      tex_array_base + b_lim(x + elem_size, y + 1, width, height, elem_size), 4,
      &Ti1j1);

  float sample = (1 - alpha) * (1 - beta) * Tij + alpha * (1 - beta) * Ti1j +
                 (1 - alpha) * beta * Tij1 + alpha * beta * Ti1j1;

  return sample;
}

float textureNormalizeElementSigned(int element, int bits) {
  if (bits) {
    int maxN = (1 << bits) - 1;
    // removing upper bits
    element &= maxN;
    // normalizing the number to [-1.0,1.0]
    maxN >>= 1;
    float output = (float)element / maxN;
    if (output < -1.0f) output = -1.0f;
    return output;
  } else {
    return 0.0f;
  }
}

float textureNormalizeElementUnsigned(unsigned int element, int bits) {
  if (bits) {
    unsigned int maxN = (1 << bits) - 1;
    // removing upper bits and normalizing the number to [0.0,1.0]
    return (float)(element & maxN) / maxN;
  } else {
    return 0.0f;
  }
}

void textureNormalizeOutput(const struct cudaChannelFormatDesc &desc,
                            ptx_reg_t &datax, ptx_reg_t &datay,
                            ptx_reg_t &dataz, ptx_reg_t &dataw) {
  if (desc.f == cudaChannelFormatKindSigned) {
    datax.f32 = textureNormalizeElementSigned(datax.s32, desc.x);
    datay.f32 = textureNormalizeElementSigned(datay.s32, desc.y);
    dataz.f32 = textureNormalizeElementSigned(dataz.s32, desc.z);
    dataw.f32 = textureNormalizeElementSigned(dataw.s32, desc.w);
  } else if (desc.f == cudaChannelFormatKindUnsigned) {
    datax.f32 = textureNormalizeElementUnsigned(datax.u32, desc.x);
    datay.f32 = textureNormalizeElementUnsigned(datay.u32, desc.y);
    dataz.f32 = textureNormalizeElementUnsigned(dataz.u32, desc.z);
    dataw.f32 = textureNormalizeElementUnsigned(dataw.u32, desc.w);
  } else {
    assert(0 &&
           "Undefined texture read mode: cudaReadModeNormalizedFloat expect "
           "integer elements");
  }
}

void tex_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  unsigned dimension = pI->dimension();
  const operand_info &dst =
      pI->dst();  // the registers to which fetched texel will be placed
  const operand_info &src1 = pI->src1();  // the name of the texture
  const operand_info &src2 =
      pI->src2();  // the vector registers containing coordinates of the texel
                   // to be fetched

  std::string texname = src1.name();
  unsigned to_type = pI->get_type();
  unsigned c_type = pI->get_type2();
  fflush(stdout);
  ptx_reg_t data1, data2, data3, data4;
  if (!thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs)
    thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs = new ptx_reg_t[4];
  unsigned nelem = src2.get_vect_nelem();
  thread->get_vector_operand_values(
      src2, thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs,
      nelem);  // ptx_reg should be 4 entry vector type...coordinates into
               // texture
  /*
    For programs with many streams, textures can be bound and unbound
    asynchronously.  This means we need to use the kernel's "snapshot" of
    the state of the texture mappings when it was launched (so that we
    don't try to access the incorrect texture mapping if it's been updated,
    or that we don't access a mapping that has been unbound).
  */
  gpgpu_t *gpu = thread->get_gpu();
  kernel_info_t &k = thread->get_kernel();
  const struct textureReference *texref = gpu->get_texref(texname);
  const struct cudaArray *cuArray = k.get_texarray(texname);
  const struct textureInfo *texInfo = k.get_texinfo(texname);
  const struct textureReferenceAttr *texAttr = gpu->get_texattr(texname);

  // assume always 2D f32 input
  // access array with src2 coordinates
  memory_space *mem = thread->get_global_memory();
  float x_f32, y_f32;
  size_t size;
  int t;
  unsigned tex_array_base;
  unsigned int width = 0, height = 0;
  int x = 0;
  int y = 0;
  unsigned tex_array_index;
  float alpha = 0, beta = 0;

  type_info_key::type_decode(to_type, size, t);
  tex_array_base = cuArray->devPtr32;

  switch (dimension) {
    case GEOM_MODIFIER_1D:
      width = cuArray->width;
      height = cuArray->height;
      if (texref->normalized) {
        assert(c_type == F32_TYPE);
        x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          x_f32 = (x_f32 > 1.0) ? 1.0 : x_f32;
          x_f32 = (x_f32 < 0.0) ? 0.0 : x_f32;
        } else if (texref->addressMode[0] == cudaAddressModeWrap) {
          x_f32 = x_f32 - floor(x_f32);
        }

        if (texref->filterMode == cudaFilterModeLinear) {
          float xb = x_f32 * width - 0.5;
          alpha = xb - floor(xb);
          alpha = reduce_precision(alpha, 9);
          beta = 0.0;

          x = (int)floor(xb);
          y = 0;
        } else {
          x = (int)floor(x_f32 * width);
          y = 0;
        }
      } else {
        switch (c_type) {
          case S32_TYPE:
            x = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].s32;
            assert(texref->filterMode == cudaFilterModePoint);
            break;
          case F32_TYPE:
            x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
            alpha = x_f32 -
                    floor(x_f32);  // offset into subtexel (for linear sampling)
            x = (int)x_f32;
            break;
          default:
            assert(0 && "Unsupported texture coordinate type.");
        }
        // handle texture fetch that exceeded boundaries
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          x = (x > width - 1) ? (width - 1) : x;
          x = (x < 0) ? 0 : x;
        } else if (texref->addressMode[0] == cudaAddressModeWrap) {
          x = x % width;
        }
      }
      width *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
                cuArray->desc.z) /
               8;
      x *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
            cuArray->desc.z) /
           8;
      tex_array_index = tex_array_base + x;

      break;
    case GEOM_MODIFIER_2D:
      width = cuArray->width;
      height = cuArray->height;
      if (texref->normalized) {
        x_f32 = reduce_precision(
            thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32, 16);
        y_f32 = reduce_precision(
            thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[1].f32, 15);

        if (texref->addressMode[0]) {  // clamp
          if (x_f32 < 0) x_f32 = 0;
          if (x_f32 >= 1) x_f32 = 1 - 1 / x_f32;
        } else {  // wrap
          x_f32 = x_f32 - floor(x_f32);
        }
        if (texref->addressMode[1]) {  // clamp
          if (y_f32 < 0) y_f32 = 0;
          if (y_f32 >= 1) y_f32 = 1 - 1 / y_f32;
        } else {  // wrap
          y_f32 = y_f32 - floor(y_f32);
        }

        if (texref->filterMode == cudaFilterModeLinear) {
          float xb = x_f32 * width - 0.5;
          float yb = y_f32 * height - 0.5;
          alpha = xb - floor(xb);
          beta = yb - floor(yb);
          alpha = reduce_precision(alpha, 9);
          beta = reduce_precision(beta, 9);

          x = (int)floor(xb);
          y = (int)floor(yb);
        } else {
          x = (int)floor(x_f32 * width);
          y = (int)floor(y_f32 * height);
        }
      } else {
        x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
        y_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[1].f32;

        alpha = x_f32 - floor(x_f32);
        beta = y_f32 - floor(y_f32);

        x = (int)x_f32;
        y = (int)y_f32;
        if (texref->addressMode[0]) {  // clamp
          if (x < 0) x = 0;
          if (x >= (int)width) x = width - 1;
        } else {  // wrap
          x = x % width;
          if (x < 0) x *= -1;
        }
        if (texref->addressMode[1]) {  // clamp
          if (y < 0) y = 0;
          if (y >= (int)height) y = height - 1;
        } else {  // wrap
          y = y % height;
          if (y < 0) y *= -1;
        }
      }

      width *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
                cuArray->desc.z) /
               8;
      x *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
            cuArray->desc.z) /
           8;
      tex_array_index = tex_array_base + (x + width * y);
      break;
    default:
      assert(0);
      break;
  }
  switch (to_type) {
    case U8_TYPE:
    case U16_TYPE:
    case U32_TYPE:
    case B8_TYPE:
    case B16_TYPE:
    case B32_TYPE:
    case S8_TYPE:
    case S16_TYPE:
    case S32_TYPE: {
      unsigned long long elementOffset = 0;  // offset into the next element
      mem->read(tex_array_index, cuArray->desc.x / 8, &data1.u32);
      elementOffset += cuArray->desc.x / 8;
      if (cuArray->desc.y) {
        mem->read(tex_array_index + elementOffset, cuArray->desc.y / 8,
                  &data2.u32);
        elementOffset += cuArray->desc.y / 8;
        if (cuArray->desc.z) {
          mem->read(tex_array_index + elementOffset, cuArray->desc.z / 8,
                    &data3.u32);
          elementOffset += cuArray->desc.z / 8;
          if (cuArray->desc.w)
            mem->read(tex_array_index + elementOffset, cuArray->desc.w / 8,
                      &data4.u32);
        }
      }
      break;
    }
    case B64_TYPE:
    case U64_TYPE:
    case S64_TYPE:
      mem->read(tex_array_index, 8, &data1.u64);
      if (cuArray->desc.y) {
        mem->read(tex_array_index + 8, 8, &data2.u64);
        if (cuArray->desc.z) {
          mem->read(tex_array_index + 16, 8, &data3.u64);
          if (cuArray->desc.w) mem->read(tex_array_index + 24, 8, &data4.u64);
        }
      }
      break;
    case F16_TYPE:
      assert(0);
      break;
    case F32_TYPE: {
      if (texref->filterMode == cudaFilterModeLinear) {
        texAddr_t b_lim = wrap;
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          b_lim = clamp;
        }
        size_t elem_size = (cuArray->desc.x + cuArray->desc.y +
                            cuArray->desc.z + cuArray->desc.w) /
                           8;
        size_t elem_ofst = 0;

        data1.f32 =
            tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                              height, elem_size, alpha, beta, b_lim);
        elem_ofst += cuArray->desc.x / 8;
        if (cuArray->desc.y) {
          data2.f32 =
              tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                                height, elem_size, alpha, beta, b_lim);
          elem_ofst += cuArray->desc.y / 8;
          if (cuArray->desc.z) {
            data3.f32 =
                tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                                  height, elem_size, alpha, beta, b_lim);
            elem_ofst += cuArray->desc.z / 8;
            if (cuArray->desc.w)
              data4.f32 = tex_linf_sampling(mem, tex_array_base, x + elem_ofst,
                                            y, width, height, elem_size, alpha,
                                            beta, b_lim);
          }
        }
      } else {
        mem->read(tex_array_index, cuArray->desc.x / 8, &data1.f32);
        if (cuArray->desc.y) {
          mem->read(tex_array_index + 4, cuArray->desc.y / 8, &data2.f32);
          if (cuArray->desc.z) {
            mem->read(tex_array_index + 8, cuArray->desc.z / 8, &data3.f32);
            if (cuArray->desc.w)
              mem->read(tex_array_index + 12, cuArray->desc.w / 8, &data4.f32);
          }
        }
      }
    } break;
    case F64_TYPE:
    case FF64_TYPE:
      mem->read(tex_array_index, 8, &data1.f64);
      if (cuArray->desc.y) {
        mem->read(tex_array_index + 8, 8, &data2.f64);
        if (cuArray->desc.z) {
          mem->read(tex_array_index + 16, 8, &data3.f64);
          if (cuArray->desc.w) mem->read(tex_array_index + 24, 8, &data4.f64);
        }
      }
      break;
    default:
      assert(0);
      break;
  }
  int x_block_coord, y_block_coord, memreqindex, blockoffset;

  switch (dimension) {
    case GEOM_MODIFIER_1D:
      thread->m_last_effective_address = tex_array_index;
      break;
    case GEOM_MODIFIER_2D:
      x_block_coord = x >> (texInfo->Tx_numbits + texInfo->texel_size_numbits);
      y_block_coord = y >> texInfo->Ty_numbits;

      memreqindex =
          ((y_block_coord * cuArray->width / texInfo->Tx) + x_block_coord) << 6;

      blockoffset = (x % (texInfo->Tx * texInfo->texel_size) +
                     (y % (texInfo->Ty)
                      << (texInfo->Tx_numbits + texInfo->texel_size_numbits)));
      memreqindex += blockoffset;
      thread->m_last_effective_address =
          tex_array_base + memreqindex;  // tex_array_index;
      break;
    default:
      assert(0);
  }
  thread->m_last_memory_space = tex_space;

  // normalize output into floating point numbers according to the texture read
  // mode
  if (texAttr->m_readmode == cudaReadModeNormalizedFloat) {
    textureNormalizeOutput(cuArray->desc, data1, data2, data3, data4);
  } else {
    assert(texAttr->m_readmode == cudaReadModeElementType);
  }

  thread->set_vector_operand_values(dst, data1, data2, data3, data4);
}

void txq_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void trap_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vabsdiff_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vadd_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vmad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

#define VMAX 0
#define VMIN 1

void vmax_impl(const ptx_instruction *pI, ptx_thread_info *thread)
{
   video_mem_instruction(pI, thread, VMAX);
}
void vmin_impl(const ptx_instruction *pI, ptx_thread_info *thread)
{
  video_mem_instruction(pI, thread, VMIN);
}
void vset_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vshl_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vshr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}
void vsub_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void vote_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static bool first_in_warp = true;
  static bool and_all;
  static bool or_all;
  static unsigned int ballot_result;
  static std::list<ptx_thread_info *> threads_in_warp;
  static unsigned last_tid;

  if (first_in_warp) {
    first_in_warp = false;
    threads_in_warp.clear();
    and_all = true;
    or_all = false;
    ballot_result = 0;
    int offset = 31;
    while ((offset >= 0) && !pI->active(offset)) offset--;
    assert(offset >= 0);
    last_tid =
        (thread->get_hw_tid() - (thread->get_hw_tid() % pI->warp_size())) +
        offset;
  }

  ptx_reg_t src1_data;
  const operand_info &src1 = pI->src1();
  src1_data = thread->get_operand_value(src1, pI->dst(), PRED_TYPE, thread, 1);

  // predicate value was changed so the lowest bit being set means the zero flag
  // is set. As a result, the value of src1_data.pred must be inverted to get
  // proper behavior
  bool pred_value = !(src1_data.pred & 0x0001);
  bool invert = src1.is_neg_pred();

  threads_in_warp.push_back(thread);
  and_all &= (invert ^ pred_value);
  or_all |= (invert ^ pred_value);

  // vote.ballot
  if (invert ^ pred_value) {
    int lane_id = thread->get_hw_tid() % pI->warp_size();
    ballot_result |= (1 << lane_id);
  }

  if (thread->get_hw_tid() == last_tid) {
    if (pI->vote_mode() == ptx_instruction::vote_ballot) {
      ptx_reg_t data = ballot_result;
      for (std::list<ptx_thread_info *>::iterator t = threads_in_warp.begin();
           t != threads_in_warp.end(); ++t) {
        const operand_info &dst = pI->dst();
        (*t)->set_operand_value(dst, data, pI->get_type(), (*t), pI);
      }
    } else {
      bool pred_value = false;

      switch (pI->vote_mode()) {
        case ptx_instruction::vote_any:
          pred_value = or_all;
          break;
        case ptx_instruction::vote_all:
          pred_value = and_all;
          break;
        case ptx_instruction::vote_uni:
          pred_value = (or_all ^ and_all);
          break;
        default:
          abort();
      }
      ptx_reg_t data;
      data.pred = pred_value ? 0 : 1;  // the way ptxplus handles the zero flag,
                                       // 1 = false and 0 = true

      for (std::list<ptx_thread_info *>::iterator t = threads_in_warp.begin();
           t != threads_in_warp.end(); ++t) {
        const operand_info &dst = pI->dst();
        (*t)->set_operand_value(dst, data, PRED_TYPE, (*t), pI);
      }
    }
    first_in_warp = true;
  }
}

void activemask_impl( const ptx_instruction *pI, ptx_thread_info *thread )
{
  active_mask_t l_activemask_bitset = pI->get_warp_active_mask();
  uint32_t l_activemask_uint = static_cast<uint32_t>(l_activemask_bitset.to_ulong());

  const operand_info &dst  = pI->dst();
  thread->set_operand_value(dst, l_activemask_uint, U32_TYPE, thread, pI);
}

void xor_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    data.pred = ~(~(src1_data.pred) ^ ~(src2_data.pred));
  else
    data.u64 = src1_data.u64 ^ src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void inst_not_implemented(const ptx_instruction *pI) {
  printf(
      "GPGPU-Sim PTX: ERROR (%s:%u) instruction \"%s\" not (yet) implemented\n",
      pI->source_file(), pI->source_line(), pI->get_opcode_cstr());
  abort();
}

ptx_reg_t srcOperandModifiers(ptx_reg_t opData, operand_info opInfo,
                              operand_info dstInfo, unsigned type,
                              ptx_thread_info *thread) {
  ptx_reg_t result;
  memory_space *mem = NULL;
  size_t size;
  int t;
  result.u64 = 0;

  // complete other cases for reading from memory, such as reading from other
  // const memory
  if (opInfo.get_addr_space() == global_space) {
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);
    mem->read(opData.u32, size / 8, &result.u64);
    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);
  } else if (opInfo.get_addr_space() == shared_space) {
    mem = thread->m_shared_mem;
    type_info_key::type_decode(type, size, t);
    mem->read(opData.u32, size / 8, &result.u64);

    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);

  } else if (opInfo.get_addr_space() == const_space) {
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);

    mem->read((opData.u32 + opInfo.get_const_mem_offset()), size / 8,
              &result.u64);

    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);
  } else {
    result = opData;
  }

  if (opInfo.get_operand_lohi() == 1) {
    result.u64 = result.u64 & 0xFFFF;
  } else if (opInfo.get_operand_lohi() == 2) {
    result.u64 = (result.u64 >> 16) & 0xFFFF;
  }

  if (opInfo.get_operand_neg() == true) {
    result.f32 = -result.f32;
  }

  return result;
}

void video_mem_instruction(const ptx_instruction *pI, ptx_thread_info *thread, int op_code)
{
  const operand_info &dst  = pI->dst(); // d
  const operand_info &src1 = pI->src1(); // a
  const operand_info &src2 = pI->src2(); // b
  const operand_info &src3 = pI->src3(); // c

  const unsigned i_type = pI->get_type();

  std::list<int> scalar_type;
  std::list<int> options;

  ptx_reg_t a, b, ta, tb, c, data;

  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  // TODO: implement this
  // ta = partSelectSignExtend( a, atype );
  // tb = partSelectSignExtend( b, btype );
  ta = a;
  tb = b;

  options = pI->get_options();
  assert(options.size() == 1);

  auto option = options.begin();
  assert(*option == ATOMIC_MAX || *option == ATOMIC_MIN);

  switch ( i_type ) {
    case S32_TYPE: {
      // assert all operands are S32_TYPE:
      scalar_type = pI->get_scalar_type();
      for (std::list<int>::iterator scalar = scalar_type.begin(); scalar != scalar_type.end(); scalar++)
      {
        assert(*scalar == S32_TYPE);
      }
      assert(scalar_type.size() == 3);
      scalar_type.clear();

      switch (op_code)
      {
        case VMAX:
          data.s32 = MY_MAX_I(ta.s32, tb.s32);
          break;
        case VMIN:
          data.s32 = MY_MIN_I(ta.s32, tb.s32);
          break;
        default:
          assert(0);
      }

      switch (*option)
      {
        case ATOMIC_MAX:
          data.s32 = MY_MAX_I(data.s32, c.s32);
        break;
        case ATOMIC_MIN:
          data.s32 = MY_MIN_I(data.s32, c.s32);
        break;
        default:
          assert(0); // not yet implemented
      }
      break;

    }
    default:
      assert(0); // not yet implemented
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);

  return;
}

void load_ray_launch_id_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 3);
  const operand_info &src0 = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  uint32_t v[4];
  v[0] = thread->get_tid().x + thread->get_ctaid().x * 32;
  v[1] = thread->get_ctaid().y;
  v[2] = thread->get_ctaid().z;

  // v[0] = 0;
  // v[1] = 30;
  // v[2] = 0;

  ptx_reg_t data;
  data.u32 = v[0];
  thread->set_operand_value(src0, data, U32_TYPE, thread, pI);

  data.u32 = v[1];
  thread->set_operand_value(src1, data, U32_TYPE, thread, pI);

  data.u32 = v[2];
  thread->set_operand_value(src2, data, U32_TYPE, thread, pI);
}

void load_ray_launch_size_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 3);
  const operand_info &src0 = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  uint32_t v[4];
  v[0] = thread->get_kernel().vulkan_metadata.launch_width;
  v[1] = thread->get_kernel().vulkan_metadata.launch_height;
  v[2] = thread->get_kernel().vulkan_metadata.launch_depth;

  // v[0] = 32;
  // v[1] = 32;
  // v[2] = 1;

  ptx_reg_t data;
  data.u32 = v[0];
  thread->set_operand_value(src0, data, U32_TYPE, thread, pI);

  data.u32 = v[1];
  thread->set_operand_value(src1, data, U32_TYPE, thread, pI);

  data.u32 = v[2];
  thread->set_operand_value(src2, data, U32_TYPE, thread, pI);
}

void load_ray_instance_custom_index_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

  uint32_t shader_counter;
  mem->read(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter);

  uint32_t shader_type;
  mem->read(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &shader_type);

  uint32_t instance_index;
  if(shader_counter == -1) // not in intersection shader or anyhit shader
    mem->read(&(traversal_data->closest_hit.instance_index), sizeof(traversal_data->closest_hit.instance_index), &instance_index);
  else {
    // intersection shader
    if (shader_type == 1) {
      warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
      instance_index = table->get_instanceID(shader_counter, thread->get_tid().x, pI, thread);
    }
    else if (shader_type == 2) {
      warp_intersection_table* table = VulkanRayTracing::anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
      instance_index = table->get_instanceID(shader_counter, thread->get_tid().x, pI, thread);
    }
    else {
      printf("Unrecognized shader_type %d\n", shader_type);
      abort();
    }
  }

  assert(pI->get_num_operands() == 1);
  const operand_info &dst = pI->dst();

  ptx_reg_t data;
  data.u32 = instance_index;
  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

void load_primitive_id_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

  uint32_t shader_counter;
  mem->read(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter);

  uint32_t shader_type;
  mem->read(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &shader_type);

  uint32_t primitive_index;
  if(shader_counter == -1) // not in intersection shader
    mem->read(&(traversal_data->closest_hit.primitive_index), sizeof(traversal_data->closest_hit.primitive_index), &primitive_index);
  else {
    if (shader_type == 1) {
      warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
      primitive_index = table->get_primitiveID(shader_counter, thread->get_tid().x, pI, thread);
    }
    else if (shader_type == 2) {
      warp_intersection_table* table = VulkanRayTracing::anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
      primitive_index = table->get_primitiveID(shader_counter, thread->get_tid().x, pI, thread);
    }
    else {
      printf("Unrecognized shader_type %d\n", shader_type);
      abort();
    }
  }


  assert(pI->get_num_operands() == 1);
  const operand_info &dst = pI->dst();

  ptx_reg_t data;
  data.u32 = primitive_index;
  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

void load_ray_world_to_object_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 2);
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();

  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, dst, U32_TYPE, thread, 1);

  data.u64 = (uint64_t)thread->RT_thread_data->traversal_data.back()->closest_hit.worldToObjectMatrix.m[src_data.u32];

  thread->set_operand_value(dst, data, B64_TYPE, thread, pI);
}

void load_ray_object_to_world_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 2);
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();

  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, dst, U32_TYPE, thread, 1);

  data.u64 = (uint64_t)thread->RT_thread_data->traversal_data.back()->closest_hit.objectToWorldMatrix.m[src_data.u32];

  thread->set_operand_value(dst, data, B64_TYPE, thread, pI);
}

void load_ray_world_direction_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 1);
  const operand_info &dst = pI->dst();

  ptx_reg_t data;
  data.u64 = (uint64_t)(&thread->RT_thread_data->traversal_data.back()->ray_world_direction.x);
  thread->set_operand_value(dst, data, B64_TYPE, thread, pI);
}

void load_ray_world_origin_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 1);
  const operand_info &dst = pI->dst();

  ptx_reg_t data;
  data.u64 = (uint64_t)(&thread->RT_thread_data->traversal_data.back()->ray_world_origin.x);
  thread->set_operand_value(dst, data, B64_TYPE, thread, pI);
}

void load_ray_t_max_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst0 = pI->dst();

  float t_max;

  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

  bool hit_geometry;
  mem->read(&(traversal_data->hit_geometry), sizeof(traversal_data->hit_geometry), &hit_geometry);
  if(hit_geometry)
    mem->read(&(traversal_data->closest_hit.world_min_thit), sizeof(t_max), &t_max);
  else
    mem->read(&(traversal_data->Tmax), sizeof(t_max), &t_max);

  ptx_reg_t data;
  data.f32 = t_max;
  thread->set_operand_value(dst0, data, F32_TYPE, thread, pI);
}

void load_ray_t_min_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

  const operand_info &dst0 = pI->dst();

  ptx_reg_t data;
  mem->read(&(traversal_data->Tmin), sizeof(data.f32), &(data.f32));
  thread->set_operand_value(dst0, data, F32_TYPE, thread, pI);
}

void vulkan_resource_index_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void load_vulkan_descriptor_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: load_vulkan_descriptor implementation\n");

  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  src1_data = thread->get_operand_value(src1, dst, U32_TYPE, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, U32_TYPE, thread, 1);

  data.u64 = (uint64_t)(VulkanRayTracing::getDescriptorAddress(src1_data.u32, src2_data.u32));
  thread->set_operand_value(dst, data, B64_TYPE, thread, pI);
}

void txl_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  TXL_DPRINTF("Functional simulation of txl instruction.\n");
  
  ptx_reg_t src0_data, src1_data, src6_data, src7_data, src8_data, data;
  
  const operand_info &src0 = pI->operand_lookup(0);
  src0_data = thread->get_operand_value(src0, src0, U64_TYPE, thread, 1);
  void* desc = (void*)(src0_data.u64);

  const operand_info &src1 = pI->operand_lookup(1);
  src1_data = thread->get_operand_value(src1, src1, U64_TYPE, thread, 1);
  void* sampler = (void*)(src1_data.u64);

  const operand_info &src6 = pI->operand_lookup(6);
  src6_data = thread->get_operand_value(src6, src6, F32_TYPE, thread, 1);
  float x = src6_data.f32;

  const operand_info &src7 = pI->operand_lookup(7);
  src7_data = thread->get_operand_value(src7, src7, F32_TYPE, thread, 1);
  float y = src7_data.f32;

  const operand_info &src8 = pI->operand_lookup(8);
  src8_data = thread->get_operand_value(src8, src8, F32_TYPE, thread, 1);
  float lod = src8_data.f32;

  float c0, c1, c2, c3;

  ptx_reg_t offset_reg;
  char *workload = getenv("VULKAN_SIM_LAUNCHER_WORKLOAD");
  if(workload && !strcmp(workload, "raytracing_extended"))
  {
    std::string reg_name("%ssa_317_array_index_64");
    offset_reg = thread->get_reg(reg_name);
    desc = desc - offset_reg.u64;
  }

  std::vector<ImageMemoryTransactionRecord> transactions;
  VulkanRayTracing::getTexture(desc, x, y, lod, c0, c1, c2, c3, transactions, offset_reg.u64);

  const operand_info &dst2 = pI->operand_lookup(2);
  const operand_info &dst3 = pI->operand_lookup(3);
  const operand_info &dst4 = pI->operand_lookup(4);
  const operand_info &dst5 = pI->operand_lookup(5);

  data.f32 = c0;
  thread->set_operand_value(dst2, data, F32_TYPE, thread, pI);

  data.f32 = c1;
  thread->set_operand_value(dst3, data, F32_TYPE, thread, pI);

  data.f32 = c2;
  thread->set_operand_value(dst4, data, F32_TYPE, thread, pI);

  data.f32 = c3;
  thread->set_operand_value(dst5, data, F32_TYPE, thread, pI);

  TXL_DPRINTF("Setting %d transactions in thread as tex_space\n", transactions.size());
  thread->set_txl_transactions(transactions);
  thread->m_last_memory_space = tex_space;
}

void ignore_ray_intersection_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: ignore_ray_intersection_impl\n");

  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

  int32_t shader_counter;
  mem->read(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter);

  Hit_data* hit_data_addr = thread->RT_thread_data->all_hit_data[shader_counter];
  Hit_data attributes;
  mem->read(hit_data_addr, sizeof(Hit_data), &attributes);

  float3 bary = attributes.barycentric_coordinates;
  VSIM_DPRINTF("gpgpusim: Ray [%d] ignoring hit %d at (%5.3f, %5.3f, %5.3f) with t = %5.3f\n", thread->get_uid(), shader_counter, bary.x, bary.y, bary.z, attributes.world_min_thit);

  // Mark world_min_thit as negative number to indicate invalid hit (all valid hit points should have t > 0)
  // traversal_data->all_hits[shader_counter].world_min_thit = -1;
  float invalid_hit = -1.0f;
  mem->write(&(hit_data_addr->world_min_thit), sizeof(float), &invalid_hit, thread, pI);
}

void report_ray_intersection_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: report_ray_intersection_impl\n");
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  src1_data = thread->get_operand_value(src1, dst, F32_TYPE, thread, 1);
  float t_hit = src1_data.f32;

  src2_data = thread->get_operand_value(src2, dst, U32_TYPE, thread, 1);
  uint32_t hit_kind = src2_data.u32;

  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

  float Tmin;
  mem->read(&(traversal_data->Tmin), sizeof(traversal_data->Tmin), &Tmin);

  bool return_value = false;

  if((Tmin <= t_hit)) {
    float Tmax;
    mem->read(&(traversal_data->Tmax), sizeof(traversal_data->Tmax), &Tmax);

    float world_min_thit;
    mem->read(&(traversal_data->closest_hit.world_min_thit), sizeof(traversal_data->closest_hit.world_min_thit), &world_min_thit);

    bool hit_geometry;
    mem->read(&(traversal_data->hit_geometry), sizeof(traversal_data->hit_geometry), &hit_geometry);

    if((hit_geometry && t_hit < world_min_thit) || (!hit_geometry && t_hit <= Tmax)) {
      int32_t shader_counter;
      mem->read(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter);

      assert(shader_counter != -1);
      warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];

      return_value = true;

      hit_geometry = true;
      mem->write(&(traversal_data->hit_geometry), sizeof(traversal_data->hit_geometry), &hit_geometry, thread, pI);

      VkGeometryTypeKHR geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
      mem->write(&(traversal_data->closest_hit.geometryType), sizeof(VkGeometryTypeKHR), &geometryType, thread, pI);

      int32_t hitGroupIndex = table->get_hitGroupIndex(shader_counter, thread->get_tid().x, pI, thread);
      mem->write(&(traversal_data->closest_hit.hitGroupIndex), sizeof(traversal_data->closest_hit.hitGroupIndex), &hitGroupIndex, thread, pI);

      mem->write(&(traversal_data->closest_hit.world_min_thit), sizeof(traversal_data->closest_hit.world_min_thit), &t_hit, thread, pI);

      uint32_t primitive_index = table->get_primitiveID(shader_counter, thread->get_tid().x, pI, thread);
      mem->write(&(traversal_data->closest_hit.primitive_index), sizeof(traversal_data->closest_hit.primitive_index), &primitive_index, thread, pI);

      uint32_t instance_index = table->get_instanceID(shader_counter, thread->get_tid().x, pI, thread);
      mem->write(&(traversal_data->closest_hit.instance_index), sizeof(traversal_data->closest_hit.instance_index), &instance_index, thread, pI);
    }
  }

  data.pred =
      (return_value ==
       0);  // inverting predicate since ptxplus uses "1" for a set zero flag
  thread->set_operand_value(dst, data, PRED_TYPE, thread, pI);
}


void deref_var_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
}

void deref_cast_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void deref_struct_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void deref_array_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void load_deref_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void trace_ray_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  //   if(thread->get_ctaid().x == 0 && thread->get_ctaid().y == 0 && thread->get_ctaid().z == 0)
    VSIM_DPRINTF("gpgpusim: trace ray implementation\n");
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }

  // const operand_info &target = pI->func_addr();
  // const symbol *func_addr = target.get_symbol();
  // function_info *target_func = func_addr->get_pc();

  // unsigned n_return = target_func->has_return();
  // assert(n_return == 0);
  // unsigned n_args = target_func->num_args();
  // assert(n_args == 11);



  int arg = 0;
  const operand_info &op1 = pI->operand_lookup(arg);
  ptx_reg_t op1_data = thread->get_operand_value(op1, op1, B64_TYPE, thread, 1);
  VkAccelerationStructureKHR _topLevelAS = (VkAccelerationStructureKHR)(op1_data.s64);

  arg++;
  const operand_info &op2 = pI->operand_lookup(arg);
  ptx_reg_t op2_data = thread->get_operand_value(op2, op2, U32_TYPE, thread, 1);
  uint rayFlags = op2_data.u32;


  arg++;
  const operand_info &op3 = pI->operand_lookup(arg);
  ptx_reg_t op3_data = thread->get_operand_value(op3, op3, U32_TYPE, thread, 1);
  uint cullMask = op3_data.u32;

  arg++;
  const operand_info &op4 = pI->operand_lookup(arg);
  ptx_reg_t op4_data = thread->get_operand_value(op4, op4, U32_TYPE, thread, 1);
  uint sbtRecordOffset = op4_data.u32;

  arg++;
  const operand_info &op5 = pI->operand_lookup(arg);
  ptx_reg_t op5_data = thread->get_operand_value(op5, op5, U32_TYPE, thread, 1);
  uint sbtRecordStride = op5_data.u32;

  arg++;
  const operand_info &op6 = pI->operand_lookup(arg);
  ptx_reg_t op6_data = thread->get_operand_value(op6, op6, U32_TYPE, thread, 1);
  uint missIndex = op6_data.u32;

  arg++;
  const operand_info &op7 = pI->operand_lookup(arg);
  ptx_reg_t op7_data = thread->get_operand_value(op7, op7, F32_TYPE, thread, 1);
  float originX = op7_data.f32;

  arg++;
  const operand_info &op8 = pI->operand_lookup(arg);
  ptx_reg_t op8_data = thread->get_operand_value(op8, op8, F32_TYPE, thread, 1);
  float originY = op8_data.f32;

  arg++;
  const operand_info &op9 = pI->operand_lookup(arg);
  ptx_reg_t op9_data = thread->get_operand_value(op9, op9, F32_TYPE, thread, 1);
  float originZ = op9_data.f32;

  arg++;
  const operand_info &op10 = pI->operand_lookup(arg);
  ptx_reg_t op10_data = thread->get_operand_value(op10, op10, F32_TYPE, thread, 1);
  float Tmin = op10_data.f32;

  arg++;
  const operand_info &op11 = pI->operand_lookup(arg);
  ptx_reg_t op11_data = thread->get_operand_value(op11, op11, F32_TYPE, thread, 1);
  float directionX = op11_data.f32;

  arg++;
  const operand_info &op12 = pI->operand_lookup(arg);
  ptx_reg_t op12_data = thread->get_operand_value(op12, op12, F32_TYPE, thread, 1);
  float directionY = op12_data.f32;

  arg++;
  const operand_info &op13 = pI->operand_lookup(arg);
  ptx_reg_t op13_data = thread->get_operand_value(op13, op13, F32_TYPE, thread, 1);
  float directionZ = op13_data.f32;

  arg++;
  const operand_info &op14 = pI->operand_lookup(arg);
  ptx_reg_t op14_data = thread->get_operand_value(op14, op14, F32_TYPE, thread, 1);
  float Tmax = op14_data.f32;

  // arg++;
  // const operand_info &op15 = pI->operand_lookup(arg);
  // ptx_reg_t op15_data = thread->get_operand_value(op15, op15, U64_TYPE, thread, 1);
  // uint32_t payload = op15_data.u64;

  // thread->dump_regs(stdout);

  VulkanRayTracing::traceRay(_topLevelAS, rayFlags, cullMask, sbtRecordOffset, sbtRecordStride, missIndex,
                   {originX, originY, originZ},
                   Tmin,
                   {directionX, directionY, directionZ},
                   Tmax,
                   NULL,
                   pI,
                   thread);
}

struct rtcore_adapter_completion_record {
  rtcore_adapter_completion_record()
      : valid(false),
        warp_uid(0),
        warp_id(0),
        owner_hw_sid(0),
        active_mask(0),
        completed_lane_mask(0),
        static_inst_uid(0),
        max_node_visits(0),
        max_primitive_tests(0) {}

  bool valid;
  unsigned warp_uid;
  unsigned warp_id;
  unsigned owner_hw_sid;
  unsigned active_mask;
  unsigned completed_lane_mask;
  unsigned static_inst_uid;
  unsigned max_node_visits;
  unsigned max_primitive_tests;
};

struct rtcore_adapter_completion_publication {
  rtcore_adapter_completion_publication()
      : warp_uid(0),
        warp_id(0),
        owner_hw_sid(0),
        active_mask(0),
        static_inst_uid(0),
        lane_slot_index(0),
        lane_thread_mask(0),
        node_visits(0),
        primitive_tests(0) {}

  unsigned warp_uid;
  unsigned warp_id;
  unsigned owner_hw_sid;
  unsigned active_mask;
  unsigned static_inst_uid;
  unsigned lane_slot_index;
  unsigned lane_thread_mask;
  unsigned node_visits;
  unsigned primitive_tests;
};

struct rtcore_adapter_completion_claim_status {
  rtcore_adapter_completion_claim_status()
      : found(false),
        metadata_match(false),
        lanes_complete(false),
        accepted(false),
        active_mask(0),
        completed_lane_mask(0),
        static_inst_uid(0),
        max_node_visits(0),
        max_primitive_tests(0),
        reject_reason("missing_record") {}

  bool found;
  bool metadata_match;
  bool lanes_complete;
  bool accepted;
  unsigned active_mask;
  unsigned completed_lane_mask;
  unsigned static_inst_uid;
  unsigned max_node_visits;
  unsigned max_primitive_tests;
  const char *reject_reason;
};

static std::map<unsigned, rtcore_adapter_completion_record>
    g_rtcore_adapter_completions;

static bool rtcore_delayed_traversal_completion_enabled() {
  const char *value = getenv("VULKAN_SIM_RTCORE_DELAYED_TRAVERSAL_COMPLETION");
  return value != NULL && *value != '\0' && strcmp(value, "0") != 0;
}

static bool rtcore_test_delayed_completion_drop_lane_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_DELAYED_COMPLETION_DROP_LANE");
  return value != NULL && *value != '\0' && strcmp(value, "0") != 0;
}

static bool rtcore_test_delayed_completion_metadata_mismatch_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_DELAYED_COMPLETION_METADATA_MISMATCH");
  return value != NULL && *value != '\0' && strcmp(value, "0") != 0;
}

static bool rtcore_test_adapter_claim_failure_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_ADAPTER_CLAIM_FAILURE");
  return value != NULL && *value != '\0' && strcmp(value, "0") != 0;
}

struct rtcore_pending_traversal_completion {
  rtcore_pending_traversal_completion()
      : valid(false),
        warp_uid(0),
        warp_id(0),
        owner_hw_sid(0),
        active_mask(0),
        static_inst_uid(0),
        lane_slot_index(0),
        lane_thread_mask(0),
        context_ptr(0),
        handoff_window_base(0),
        result_word(0),
        window_generation(0),
        completion_seq_low(0),
        resume_seq_low(0),
        window_tag(0),
        reason(0) {
    memset(&traversal_snapshot, 0, sizeof(traversal_snapshot));
  }

  bool valid;
  unsigned warp_uid;
  unsigned warp_id;
  unsigned owner_hw_sid;
  unsigned active_mask;
  unsigned static_inst_uid;
  unsigned lane_slot_index;
  unsigned lane_thread_mask;
  unsigned long long context_ptr;
  unsigned long long handoff_window_base;
  unsigned result_word;
  unsigned window_generation;
  unsigned completion_seq_low;
  unsigned resume_seq_low;
  unsigned window_tag;
  unsigned reason;
  Traversal_data traversal_snapshot;
};

static std::map<std::pair<unsigned, unsigned>,
                rtcore_pending_traversal_completion>
    g_rtcore_pending_traversal_completions;

static rtcore_adapter_completion_publication
rtcore_make_adapter_completion_publication_from_pending(
    const rtcore_pending_traversal_completion &record) {
  rtcore_adapter_completion_publication publication;
  publication.warp_uid = record.warp_uid;
  publication.warp_id = record.warp_id;
  publication.owner_hw_sid = record.owner_hw_sid;
  publication.active_mask = record.active_mask;
  publication.static_inst_uid = record.static_inst_uid;
  publication.lane_slot_index = record.lane_slot_index;
  publication.lane_thread_mask = record.lane_thread_mask;
  publication.node_visits = record.traversal_snapshot.rtcore_node_visits;
  publication.primitive_tests =
      record.traversal_snapshot.rtcore_primitive_tests;
  return publication;
}

static bool rtcore_publish_adapter_completion_record(
    const rtcore_adapter_completion_publication &publication) {
  const bool valid_lane = publication.lane_slot_index < 32;
  const bool lane_active =
      (publication.active_mask & publication.lane_thread_mask) != 0;
  rtcore_adapter_completion_record &record =
      g_rtcore_adapter_completions[publication.warp_uid];
  if (!record.valid) {
    record.valid = true;
    record.warp_uid = publication.warp_uid;
    record.warp_id = publication.warp_id;
    record.owner_hw_sid = publication.owner_hw_sid;
    record.active_mask = publication.active_mask;
    record.static_inst_uid = publication.static_inst_uid;
  }

  const bool metadata_matches =
      record.warp_uid == publication.warp_uid &&
      record.warp_id == publication.warp_id &&
      record.owner_hw_sid == publication.owner_hw_sid &&
      record.active_mask == publication.active_mask &&
      record.static_inst_uid == publication.static_inst_uid;
  if (valid_lane && lane_active && metadata_matches) {
    record.completed_lane_mask |= publication.lane_thread_mask;
    if (publication.node_visits > record.max_node_visits) {
      record.max_node_visits = publication.node_visits;
    }
    if (publication.primitive_tests > record.max_primitive_tests) {
      record.max_primitive_tests = publication.primitive_tests;
    }
  }
  const bool accepted = valid_lane && lane_active && metadata_matches;

  printf("GPGPU-Sim PTX: RT_SUBMIT adapter-completion-publish, "
         "warp_uid=%u, warp_id=%u, owner_hw_sid=%u, active_mask=0x%08x, "
         "completed_lane_mask=0x%08x, static_inst_uid=%u, "
         "lane_slot_index=%u, lane_thread_mask=0x%08x, "
         "node_visits=%u, primitive_tests=%u, max_node_visits=%u, "
         "max_primitive_tests=%u, accepted=%u\n",
         publication.warp_uid, publication.warp_id, publication.owner_hw_sid,
         publication.active_mask, record.completed_lane_mask,
         publication.static_inst_uid, publication.lane_slot_index,
         publication.lane_thread_mask, publication.node_visits,
         publication.primitive_tests,
         record.max_node_visits, record.max_primitive_tests,
         accepted ? 1 : 0);
  fflush(stdout);
  return accepted;
}

extern "C" bool rtcore_publish_adapter_completion(
    unsigned warp_uid, unsigned warp_id, unsigned owner_hw_sid,
    unsigned active_mask, unsigned static_inst_uid,
    unsigned lane_slot_index, unsigned node_visits,
    unsigned primitive_tests) {
  rtcore_adapter_completion_publication publication;
  publication.warp_uid = warp_uid;
  publication.warp_id = warp_id;
  publication.owner_hw_sid = owner_hw_sid;
  publication.active_mask = active_mask;
  publication.static_inst_uid = static_inst_uid;
  publication.lane_slot_index = lane_slot_index;
  publication.lane_thread_mask =
      lane_slot_index < 32 ? (1u << lane_slot_index) : 0;
  publication.node_visits = node_visits;
  publication.primitive_tests = primitive_tests;
  return rtcore_publish_adapter_completion_record(publication);
}

extern "C" bool rtcore_service_pending_traversal_completion(
    unsigned warp_uid, unsigned warp_id, unsigned owner_hw_sid,
    unsigned issued_active_mask) {
  if (!rtcore_delayed_traversal_completion_enabled()) {
    return false;
  }

  const bool test_drop_lane =
      rtcore_test_delayed_completion_drop_lane_enabled();
  bool serviced = false;
  bool dropped_lane = false;
  unsigned dropped_lane_mask = 0;
  unsigned published_lanes = 0;
  unsigned metadata_mismatch_lanes = 0;

  std::map<std::pair<unsigned, unsigned>,
           rtcore_pending_traversal_completion>::iterator event =
      g_rtcore_pending_traversal_completions.begin();
  while (event != g_rtcore_pending_traversal_completions.end()) {
    rtcore_pending_traversal_completion record = event->second;
    if (event->first.first != warp_uid) {
      ++event;
      continue;
    }

    serviced = true;
    const bool metadata_matches =
        record.valid && record.warp_uid == warp_uid &&
        record.warp_id == warp_id && record.owner_hw_sid == owner_hw_sid &&
        record.active_mask == issued_active_mask;
    const char *reject_reason =
        metadata_matches ? "accepted" : "metadata_mismatch";
    const bool drop_this_lane =
        test_drop_lane && !dropped_lane && metadata_matches;
    bool published = false;
    if (metadata_matches && !drop_this_lane) {
      rtcore_adapter_completion_publication publication =
          rtcore_make_adapter_completion_publication_from_pending(record);
      published = rtcore_publish_adapter_completion_record(publication);
      if (published) {
        published_lanes++;
      }
    } else if (drop_this_lane) {
      dropped_lane = true;
      dropped_lane_mask = record.lane_thread_mask;
      reject_reason = "dropped_lane";
    } else {
      metadata_mismatch_lanes++;
    }

    printf("GPGPU-Sim PTX: RT-unit delayed-completion-service, "
           "warp_uid=%u, warp_id=%u, owner_hw_sid=%u, "
           "lane_slot_index=%u, lane_thread_mask=0x%08x, "
           "metadata_match=%u, reject_reason=%s, dropped_lane=%u, "
           "published=%u\n",
           warp_uid, warp_id, owner_hw_sid, record.lane_slot_index,
           record.lane_thread_mask, metadata_matches ? 1 : 0,
           reject_reason, drop_this_lane ? 1 : 0, published ? 1 : 0);
    fflush(stdout);
    event = g_rtcore_pending_traversal_completions.erase(event);
  }

  if (serviced) {
    printf("GPGPU-Sim PTX: RT-unit delayed-completion-service-summary, "
           "warp_uid=%u, warp_id=%u, owner_hw_sid=%u, "
           "issued_active_mask=0x%08x, published_lanes=%u, "
           "dropped_lane=%u, dropped_lane_mask=0x%08x, "
           "metadata_mismatch_lanes=%u\n",
           warp_uid, warp_id, owner_hw_sid, issued_active_mask,
           published_lanes, dropped_lane ? 1 : 0, dropped_lane_mask,
           metadata_mismatch_lanes);
    fflush(stdout);
  }
  return serviced;
}

static rtcore_adapter_completion_claim_status rtcore_query_adapter_completion_status_internal(
    unsigned warp_uid, unsigned warp_id, unsigned owner_hw_sid) {
  rtcore_adapter_completion_claim_status status;
  std::map<unsigned, rtcore_adapter_completion_record>::iterator event =
      g_rtcore_adapter_completions.find(warp_uid);
  if (event == g_rtcore_adapter_completions.end()) {
    return status;
  }

  const rtcore_adapter_completion_record &record = event->second;
  status.found = true;
  status.active_mask = record.active_mask;
  status.completed_lane_mask = record.completed_lane_mask;
  status.static_inst_uid = record.static_inst_uid;
  status.max_node_visits = record.max_node_visits;
  status.max_primitive_tests = record.max_primitive_tests;
  status.metadata_match =
      record.valid && record.warp_uid == warp_uid &&
      record.warp_id == warp_id && record.owner_hw_sid == owner_hw_sid;
  status.lanes_complete =
      record.active_mask != 0 &&
      (record.completed_lane_mask & record.active_mask) == record.active_mask;
  status.accepted = status.metadata_match && status.lanes_complete;
  if (status.accepted) {
    status.reject_reason = "accepted";
  } else if (!status.metadata_match) {
    status.reject_reason = "metadata_mismatch";
  } else {
    status.reject_reason = "incomplete_lanes";
  }
  return status;
}

static void rtcore_fill_adapter_completion_claim_snapshot(
    const rtcore_adapter_completion_claim_status &status,
    rtcore_adapter_completion_claim_snapshot *snapshot) {
  if (snapshot == NULL) {
    return;
  }
  snapshot->found = status.found;
  snapshot->metadata_match = status.metadata_match;
  snapshot->lanes_complete = status.lanes_complete;
  snapshot->accepted = status.accepted;
  snapshot->active_mask = status.active_mask;
  snapshot->completed_lane_mask = status.completed_lane_mask;
  snapshot->static_inst_uid = status.static_inst_uid;
  snapshot->max_node_visits = status.max_node_visits;
  snapshot->max_primitive_tests = status.max_primitive_tests;
  snapshot->reject_reason = status.reject_reason;
}

extern "C" bool rtcore_claim_adapter_completion_snapshot(
    unsigned warp_uid, unsigned warp_id, unsigned owner_hw_sid,
    rtcore_adapter_completion_claim_snapshot *snapshot) {
  rtcore_adapter_completion_claim_status status =
      rtcore_query_adapter_completion_status_internal(
          warp_uid, warp_id, owner_hw_sid);
  rtcore_fill_adapter_completion_claim_snapshot(status, snapshot);
  if (!status.found) {
    printf("GPGPU-Sim PTX: RT-unit adapter-completion-claim, "
           "warp_uid=%u, warp_id=%u, owner_hw_sid=%u, found=0, "
           "active_mask=0x%08x, completed_lane_mask=0x%08x, "
           "static_inst_uid=%u, max_node_visits=%u, "
           "max_primitive_tests=%u, metadata_match=%u, lanes_complete=%u, "
           "accepted=%u, reject_reason=%s\n",
           warp_uid, warp_id, owner_hw_sid, status.active_mask,
           status.completed_lane_mask, status.static_inst_uid,
           status.max_node_visits, status.max_primitive_tests,
           status.metadata_match ? 1 : 0, status.lanes_complete ? 1 : 0,
           status.accepted ? 1 : 0, status.reject_reason);
    fflush(stdout);
    return false;
  }

  if (status.accepted) {
    g_rtcore_adapter_completions.erase(warp_uid);
  }

  printf("GPGPU-Sim PTX: RT-unit adapter-completion-claim, "
         "warp_uid=%u, warp_id=%u, owner_hw_sid=%u, found=1, "
         "active_mask=0x%08x, completed_lane_mask=0x%08x, "
         "static_inst_uid=%u, max_node_visits=%u, "
         "max_primitive_tests=%u, metadata_match=%u, lanes_complete=%u, "
         "accepted=%u, reject_reason=%s\n",
         warp_uid, warp_id, owner_hw_sid, status.active_mask,
         status.completed_lane_mask, status.static_inst_uid,
         status.max_node_visits, status.max_primitive_tests,
         status.metadata_match ? 1 : 0, status.lanes_complete ? 1 : 0,
         status.accepted ? 1 : 0, status.reject_reason);
  fflush(stdout);
  return status.accepted;
}

extern "C" bool rtcore_claim_adapter_completion(
    unsigned warp_uid, unsigned warp_id, unsigned owner_hw_sid,
    unsigned *active_mask, unsigned *completed_lane_mask,
    unsigned *static_inst_uid, unsigned *max_node_visits,
    unsigned *max_primitive_tests) {
  rtcore_adapter_completion_claim_snapshot snapshot;
  const bool accepted = rtcore_claim_adapter_completion_snapshot(
      warp_uid, warp_id, owner_hw_sid, &snapshot);
  if (accepted) {
    if (active_mask != NULL) {
      *active_mask = snapshot.active_mask;
    }
    if (completed_lane_mask != NULL) {
      *completed_lane_mask = snapshot.completed_lane_mask;
    }
    if (static_inst_uid != NULL) {
      *static_inst_uid = snapshot.static_inst_uid;
    }
    if (max_node_visits != NULL) {
      *max_node_visits = snapshot.max_node_visits;
    }
    if (max_primitive_tests != NULL) {
      *max_primitive_tests = snapshot.max_primitive_tests;
    }
  }
  return accepted;
}

namespace {

const unsigned long long RTCORE_CONTEXT_ALIGNMENT = 64;
const unsigned long long RTCORE_HANDOFF_WINDOW_ALIGNMENT = 128;
const unsigned RTCORE_RETURN_FOR_CLOSEST_HIT = 0x02;
const unsigned RTCORE_RETURN_FOR_MISS = 0x03;
const unsigned RTCORE_RETURN_FOR_MEMORY_FAULT = 0x04;
const unsigned RTCORE_COMPLETION_FLAG_TRACE_DONE = 1u << 3;
const unsigned RTCORE_COMPLETION_FLAG_MEMORY_FAULT = 1u << 4;
const unsigned RTCORE_COMPLETION_VALID = 1u << 31;
const unsigned RTCORE_WINDOW_GROUP_VALID = RTCORE_COMPLETION_VALID;
const unsigned RTCORE_FAULT_PAYLOAD_MAGIC = 0x46544c54;
const unsigned RTCORE_FAULT_CODE_TRANSLATION = 1;
const unsigned RTCORE_MAX_LANES_PER_WARP = 32;
const unsigned RTCORE_CONTEXT_BYTES_PER_LANE = 0x280;
const unsigned RTCORE_HANDOFF_WINDOW_WORDS_PER_LANE = 32;
const unsigned RTCORE_HANDOFF_WINDOW_BYTES_PER_LANE = 128;
const unsigned RTCORE_HANDOFF_WINDOW_BYTES_PER_FULL_WARP =
    RTCORE_MAX_LANES_PER_WARP * RTCORE_HANDOFF_WINDOW_BYTES_PER_LANE;
const unsigned RTCORE_SHARED_MEMORY_BYTES_PER_EXECUTION_PARTITION = 49152;
const unsigned RTCORE_MAX_RESIDENT_WARPS_PER_EXECUTION_PARTITION = 16;
const unsigned RTCORE_RT_TOKENS_PER_EXECUTION_PARTITION = 128;
const unsigned RTCORE_TRAVERSAL_STACK_ENTRIES_PER_ACTIVE_LANE = 16;
const unsigned RTCORE_TRAVERSAL_STACK_ENTRIES_PER_EXECUTION_PARTITION =
    RTCORE_MAX_RESIDENT_WARPS_PER_EXECUTION_PARTITION *
    RTCORE_MAX_LANES_PER_WARP *
    RTCORE_TRAVERSAL_STACK_ENTRIES_PER_ACTIVE_LANE;
const unsigned RTCORE_WINDOW_STATE_COMPLETE = 3;
const unsigned long long RTCORE_SHARED_HANDOFF_WINDOW_BASE = 0x20000000ull;
const unsigned long long RTCORE_SHARED_HANDOFF_WINDOW_BYTES = 0x10000000ull;
const unsigned long long RTCORE_LOCAL_PRIVATE_HANDOFF_WINDOW_BASE =
    0x30000000ull;
const unsigned long long RTCORE_LOCAL_PRIVATE_HANDOFF_WINDOW_BYTES =
    0x10000000ull;

enum rtcore_handoff_window_class {
  RTCORE_HANDOFF_WINDOW_CLASS_SHARED_WORKGROUP,
  RTCORE_HANDOFF_WINDOW_CLASS_LOCAL_PRIVATE,
  RTCORE_HANDOFF_WINDOW_CLASS_UNSUPPORTED
};

struct rtcore_symbolic_resource_profile {
  unsigned shared_memory_bytes_available_per_execution_partition;
  unsigned max_resident_warps_per_execution_partition;
  unsigned max_active_lanes_per_rt_warp;
  unsigned handoff_window_bytes_per_active_rt_lane;
  unsigned rt_tokens_per_execution_partition;
  unsigned traversal_stack_entries_per_active_lane;
  unsigned traversal_stack_entries_per_execution_partition;
  unsigned traversal_stack_entries_per_full_warp;
  unsigned full_warp_handoff_window_bytes;
  unsigned handoff_window_limited_warps;
  unsigned token_limited_warps;
  unsigned traversal_stack_limited_warps;
  unsigned resident_rt_warps;
};

struct rtcore_synthetic_handoff_header {
  unsigned long long context_ptr;
  unsigned long long handoff_window_base;
  unsigned lane_slot_index;
  unsigned long long lane_slot_base;
  unsigned owner_hw_tid;
  unsigned owner_hw_wid;
  unsigned owner_hw_sid;
  unsigned thread_mask;
  unsigned window_state;
  unsigned window_generation;
  unsigned completion_seq;
  unsigned resume_seq;
  unsigned window_tag;
  unsigned w0;
  unsigned w1;
  unsigned w2;
  unsigned w3;
  unsigned w4;
  unsigned w5;
  unsigned w6;
  unsigned w7;
  unsigned w8;
  unsigned w9;
  unsigned w10;
  unsigned w11;
  unsigned w12;
  unsigned w13;
  unsigned w14;
  unsigned w15;
  unsigned w16;
  unsigned w17;
  unsigned w18;
  unsigned w19;
  unsigned w20;
  unsigned w21;
  unsigned w22;
  unsigned w23;
};

struct rtcore_synthetic_handoff_key {
  unsigned long long handoff_window_base;
  unsigned owner_hw_sid;
  unsigned owner_hw_wid;
  unsigned lane_slot_index;

  bool operator<(const rtcore_synthetic_handoff_key &other) const {
    if (handoff_window_base != other.handoff_window_base) {
      return handoff_window_base < other.handoff_window_base;
    }
    if (owner_hw_sid != other.owner_hw_sid) {
      return owner_hw_sid < other.owner_hw_sid;
    }
    if (owner_hw_wid != other.owner_hw_wid) {
      return owner_hw_wid < other.owner_hw_wid;
    }
    return lane_slot_index < other.lane_slot_index;
  }
};

struct rtcore_symbolic_rt_token_key {
  unsigned long long context_ptr;
  unsigned long long handoff_window_base;
  unsigned owner_hw_sid;
  unsigned owner_hw_wid;
  unsigned owner_hw_tid;
  unsigned lane_slot_index;

  bool operator<(const rtcore_symbolic_rt_token_key &other) const {
    if (context_ptr != other.context_ptr) {
      return context_ptr < other.context_ptr;
    }
    if (handoff_window_base != other.handoff_window_base) {
      return handoff_window_base < other.handoff_window_base;
    }
    if (owner_hw_sid != other.owner_hw_sid) {
      return owner_hw_sid < other.owner_hw_sid;
    }
    if (owner_hw_wid != other.owner_hw_wid) {
      return owner_hw_wid < other.owner_hw_wid;
    }
    if (owner_hw_tid != other.owner_hw_tid) {
      return owner_hw_tid < other.owner_hw_tid;
    }
    return lane_slot_index < other.lane_slot_index;
  }
};

struct rtcore_symbolic_rt_token_allocation {
  unsigned slot_id;
  unsigned generation;
  bool allocated;
};

struct rtcore_symbolic_rt_token_allocator_slot {
  unsigned slot_id;
  unsigned generation;
  bool live;
};

struct rtcore_symbolic_rt_token_allocator_state {
  unsigned next_slot_id;
  unsigned next_generation;
  size_t live_allocations;
  size_t released_allocations;
  std::map<unsigned, rtcore_symbolic_rt_token_allocator_slot> slots;
  std::vector<unsigned> free_slot_ids;
};

struct rtcore_symbolic_rt_token_record {
  unsigned token_id;
  unsigned allocator_slot_id;
  unsigned allocator_generation;
  unsigned window_generation;
  unsigned completion_seq;
  unsigned resume_seq;
  unsigned window_tag;
  unsigned result_word;
  bool completed;
};

struct rtcore_symbolic_rt_token_reservation_key {
  unsigned warp_uid;
  unsigned warp_id;
  unsigned owner_hw_sid;
  unsigned static_inst_uid;
  unsigned active_mask;
  unsigned long long context_ptr;
  unsigned long long handoff_window_base;

  bool operator<(
      const rtcore_symbolic_rt_token_reservation_key &other) const {
    if (warp_uid != other.warp_uid) {
      return warp_uid < other.warp_uid;
    }
    if (warp_id != other.warp_id) {
      return warp_id < other.warp_id;
    }
    if (owner_hw_sid != other.owner_hw_sid) {
      return owner_hw_sid < other.owner_hw_sid;
    }
    if (static_inst_uid != other.static_inst_uid) {
      return static_inst_uid < other.static_inst_uid;
    }
    if (active_mask != other.active_mask) {
      return active_mask < other.active_mask;
    }
    return handoff_window_base < other.handoff_window_base;
  }
};

struct rtcore_symbolic_rt_token_reservation_record {
  unsigned active_mask;
  unsigned active_lanes;
  unsigned acquired_lane_mask;
};

static std::map<rtcore_synthetic_handoff_key, rtcore_synthetic_handoff_header>
    g_rtcore_synthetic_handoff_windows;
static std::map<rtcore_synthetic_handoff_key, unsigned>
    g_rtcore_synthetic_window_generations;
static std::map<rtcore_symbolic_rt_token_key,
                rtcore_symbolic_rt_token_record>
    g_rtcore_symbolic_rt_tokens;
static std::map<rtcore_symbolic_rt_token_reservation_key,
                rtcore_symbolic_rt_token_reservation_record>
    g_rtcore_symbolic_rt_token_reservations;
static rtcore_symbolic_rt_token_allocator_state
    g_rtcore_symbolic_rt_token_allocator = {1, 1, 0, 0};
unsigned g_rtcore_next_symbolic_rt_token_id = 1;
bool g_rtcore_symbolic_resource_profile_logged = false;

unsigned rtcore_lane_slot_index(ptx_thread_info *thread) {
  return thread->get_hw_tid() % RTCORE_MAX_LANES_PER_WARP;
}

unsigned rtcore_handoff_lane_slot_byte_offset(unsigned lane_slot_index) {
  return lane_slot_index * RTCORE_HANDOFF_WINDOW_BYTES_PER_LANE;
}

unsigned long long rtcore_handoff_lane_slot_base(
    unsigned long long handoff_window_base, unsigned lane_slot_index) {
  return handoff_window_base +
         rtcore_handoff_lane_slot_byte_offset(lane_slot_index);
}

unsigned rtcore_lane_thread_mask(unsigned lane_slot_index) {
  return 1u << lane_slot_index;
}

unsigned rtcore_active_thread_mask(const ptx_instruction *pI) {
  active_mask_t active_mask = pI->get_warp_active_mask();
  const unsigned active_mask_value =
      static_cast<unsigned>(active_mask.to_ulong());
  if (active_mask_value != 0) {
    return active_mask_value;
  }
  return 0xffffffffu;
}

bool rtcore_test_inactive_lane_submit_enabled() {
  const char *value = getenv("VULKAN_SIM_RTCORE_TEST_INACTIVE_LANE_SUBMIT");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_test_inactive_lane_retire_enabled() {
  const char *value = getenv("VULKAN_SIM_RTCORE_TEST_INACTIVE_LANE_RETIRE");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_lane_is_active_in_mask(unsigned active_thread_mask,
                                   unsigned lane_slot_index) {
  const unsigned lane_thread_mask =
      rtcore_lane_thread_mask(lane_slot_index);
  return (active_thread_mask & lane_thread_mask) != 0;
}

bool rtcore_lane_is_active_in_instruction(const ptx_instruction *pI,
                                          unsigned lane_slot_index) {
  return rtcore_lane_is_active_in_mask(rtcore_active_thread_mask(pI),
                                       lane_slot_index);
}

unsigned rtcore_count_active_lanes(unsigned active_mask) {
  unsigned count = 0;
  for (unsigned lane = 0; lane < RTCORE_MAX_LANES_PER_WARP; ++lane) {
    if ((active_mask & (1u << lane)) != 0) {
      ++count;
    }
  }
  return count;
}

unsigned rtcore_min_u32(unsigned a, unsigned b) { return a < b ? a : b; }

unsigned rtcore_u64_to_u32_saturating(unsigned long long value) {
  return value > 0xffffffffull ? 0xffffffffu : static_cast<unsigned>(value);
}

unsigned rtcore_peek_next_synthetic_window_generation(
    const rtcore_synthetic_handoff_key &key) {
  std::map<rtcore_synthetic_handoff_key, unsigned>::const_iterator
      window_generation = g_rtcore_synthetic_window_generations.find(key);
  if (window_generation == g_rtcore_synthetic_window_generations.end()) {
    return 1;
  }
  return window_generation->second + 1;
}

void rtcore_commit_synthetic_window_generation(
    const rtcore_synthetic_handoff_key &key, unsigned window_generation) {
  g_rtcore_synthetic_window_generations[key] = window_generation;
}

unsigned rtcore_synthetic_completion_seq_for_generation(
    unsigned generation) {
  return generation & 0xffffu;
}

unsigned rtcore_synthetic_window_tag_for_generation(unsigned generation) {
  return generation & 0xffu;
}

unsigned rtcore_resource_env_u32(const char *name, unsigned fallback) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return fallback;
  }
  char *end = NULL;
  const unsigned long parsed = strtoul(value, &end, 0);
  if (end == value || *end != '\0' || parsed > 0xfffffffful) {
    return fallback;
  }
  return static_cast<unsigned>(parsed);
}

bool rtcore_test_retire_before_completion_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_RETIRE_BEFORE_COMPLETION");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_test_memory_fault_publication_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_MEMORY_FAULT_PUBLICATION");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_test_memory_fault_payload_omission_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_MEMORY_FAULT_PAYLOAD_OMISSION");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_test_memory_fault_header_omission_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_MEMORY_FAULT_HEADER_OMISSION");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_test_fail_after_token_acquire_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_FAIL_AFTER_TOKEN_ACQUIRE");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_test_fail_after_handoff_publish_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_FAIL_AFTER_HANDOFF_PUBLISH");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_test_fail_after_result_write_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_FAIL_AFTER_RESULT_WRITE");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_test_stale_allocator_generation_enabled() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TEST_STALE_ALLOCATOR_GENERATION");
  return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool rtcore_address_in_range(unsigned long long address,
                             unsigned long long base,
                             unsigned long long bytes) {
  return address >= base && address < (base + bytes);
}

rtcore_handoff_window_class rtcore_classify_handoff_window(
    unsigned long long handoff_window_base) {
  if (rtcore_address_in_range(handoff_window_base,
                              RTCORE_SHARED_HANDOFF_WINDOW_BASE,
                              RTCORE_SHARED_HANDOFF_WINDOW_BYTES)) {
    return RTCORE_HANDOFF_WINDOW_CLASS_SHARED_WORKGROUP;
  }
  if (rtcore_address_in_range(handoff_window_base,
                              RTCORE_LOCAL_PRIVATE_HANDOFF_WINDOW_BASE,
                              RTCORE_LOCAL_PRIVATE_HANDOFF_WINDOW_BYTES)) {
    return RTCORE_HANDOFF_WINDOW_CLASS_LOCAL_PRIVATE;
  }
  return RTCORE_HANDOFF_WINDOW_CLASS_UNSUPPORTED;
}

const char *rtcore_handoff_window_class_name(
    rtcore_handoff_window_class window_class) {
  switch (window_class) {
    case RTCORE_HANDOFF_WINDOW_CLASS_SHARED_WORKGROUP:
      return "shared_workgroup";
    case RTCORE_HANDOFF_WINDOW_CLASS_LOCAL_PRIVATE:
      return "local_private";
    default:
      return "unsupported";
  }
}

bool rtcore_submit_handoff_window_class_is_valid(
    const ptx_instruction *pI, unsigned long long context_ptr,
    unsigned long long handoff_window_base, unsigned lane_slot_index) {
  const rtcore_handoff_window_class window_class =
      rtcore_classify_handoff_window(handoff_window_base);
  const bool shared_workgroup =
      window_class == RTCORE_HANDOFF_WINDOW_CLASS_SHARED_WORKGROUP;
  const bool accepted = shared_workgroup;

  printf("GPGPU-Sim PTX: RT_SUBMIT handoff-window-class (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, window_class=%s, shared_workgroup=%u, "
         "accepted=%u\n",
         pI->source_file(), pI->source_line(), context_ptr,
         handoff_window_base, lane_slot_index,
         rtcore_handoff_window_class_name(window_class),
         shared_workgroup ? 1 : 0, accepted ? 1 : 0);
  fflush(stdout);

  if (!accepted) {
    const char *reason =
        window_class == RTCORE_HANDOFF_WINDOW_CLASS_LOCAL_PRIVATE
            ? "LOCAL_PRIVATE_HOT_WINDOW"
            : "UNSUPPORTED_WINDOW_CLASS";
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=%s, context_ptr=0x%llx, handoff_window_base=0x%llx, "
           "lane_slot_index=%u, window_class=%s, consumed=0\n",
           pI->source_file(), pI->source_line(), reason, context_ptr,
           handoff_window_base, lane_slot_index,
           rtcore_handoff_window_class_name(window_class));
    fflush(stdout);
  }

  return accepted;
}

rtcore_symbolic_resource_profile rtcore_get_symbolic_resource_profile() {
  rtcore_symbolic_resource_profile profile;
  profile.shared_memory_bytes_available_per_execution_partition =
      rtcore_resource_env_u32(
          "VULKAN_SIM_RTCORE_SHARED_MEMORY_BYTES",
          RTCORE_SHARED_MEMORY_BYTES_PER_EXECUTION_PARTITION);
  profile.max_resident_warps_per_execution_partition =
      rtcore_resource_env_u32(
          "VULKAN_SIM_RTCORE_MAX_RESIDENT_WARPS",
          RTCORE_MAX_RESIDENT_WARPS_PER_EXECUTION_PARTITION);
  profile.max_active_lanes_per_rt_warp = RTCORE_MAX_LANES_PER_WARP;
  profile.handoff_window_bytes_per_active_rt_lane =
      RTCORE_HANDOFF_WINDOW_BYTES_PER_LANE;
  profile.rt_tokens_per_execution_partition = rtcore_resource_env_u32(
      "VULKAN_SIM_RTCORE_RT_TOKENS",
      RTCORE_RT_TOKENS_PER_EXECUTION_PARTITION);
  profile.traversal_stack_entries_per_active_lane = rtcore_resource_env_u32(
      "VULKAN_SIM_RTCORE_TRAVERSAL_STACK_ENTRIES",
      RTCORE_TRAVERSAL_STACK_ENTRIES_PER_ACTIVE_LANE);
  profile.traversal_stack_entries_per_execution_partition =
      rtcore_resource_env_u32(
          "VULKAN_SIM_RTCORE_TRAVERSAL_STACK_ENTRIES_TOTAL",
          RTCORE_TRAVERSAL_STACK_ENTRIES_PER_EXECUTION_PARTITION);
  profile.full_warp_handoff_window_bytes =
      RTCORE_MAX_LANES_PER_WARP * RTCORE_HANDOFF_WINDOW_BYTES_PER_LANE;
  profile.traversal_stack_entries_per_full_warp =
      rtcore_u64_to_u32_saturating(
          (unsigned long long)profile.max_active_lanes_per_rt_warp *
          profile.traversal_stack_entries_per_active_lane);
  profile.handoff_window_limited_warps =
      profile.full_warp_handoff_window_bytes == 0
          ? 0
          : profile.shared_memory_bytes_available_per_execution_partition /
                profile.full_warp_handoff_window_bytes;
  profile.token_limited_warps =
      profile.max_active_lanes_per_rt_warp == 0
          ? 0
          : profile.rt_tokens_per_execution_partition /
                profile.max_active_lanes_per_rt_warp;
  profile.traversal_stack_limited_warps =
      profile.traversal_stack_entries_per_full_warp == 0
          ? 0
          : profile.traversal_stack_entries_per_execution_partition /
                profile.traversal_stack_entries_per_full_warp;
  profile.resident_rt_warps = rtcore_min_u32(
      profile.max_resident_warps_per_execution_partition,
      rtcore_min_u32(profile.handoff_window_limited_warps,
                     rtcore_min_u32(profile.token_limited_warps,
                                    profile.traversal_stack_limited_warps)));
  return profile;
}

const char *rtcore_symbolic_resource_limit_name(
    const rtcore_symbolic_resource_profile &profile) {
  if (profile.handoff_window_limited_warps == 0) {
    return "HANDOFF_WINDOW_EXHAUSTED";
  }
  if (profile.token_limited_warps == 0) {
    return "RT_TOKEN_EXHAUSTED";
  }
  if (profile.traversal_stack_limited_warps == 0) {
    return "TRAVERSAL_STACK_EXHAUSTED";
  }
  if (profile.max_resident_warps_per_execution_partition == 0) {
    return "MAX_RESIDENT_WARPS_ZERO";
  }
  return "RESOURCE_EXHAUSTED";
}

void rtcore_log_symbolic_resource_profile_once(
    const rtcore_symbolic_resource_profile &profile) {
  if (g_rtcore_symbolic_resource_profile_logged) {
    return;
  }
  g_rtcore_symbolic_resource_profile_logged = true;

  printf("GPGPU-Sim PTX: RT resource-profile "
         "lanes_per_warp=%u, handoff_words_per_lane=%u, "
         "handoff_bytes_per_lane=%u, handoff_bytes_full_warp=%u, "
         "context_bytes_per_lane=%u, "
         "shared_memory_bytes_per_execution_partition=%u, "
         "max_resident_warps_per_execution_partition=%u, "
         "max_active_lanes_per_rt_warp=%u, "
         "handoff_window_bytes_per_active_rt_lane=%u, "
         "rt_tokens_per_execution_partition=%u, "
         "traversal_stack_entries_per_active_lane=%u, "
         "traversal_stack_entries_per_execution_partition=%u, "
         "traversal_stack_entries_per_full_warp=%u, "
         "handoff_window_limited_warps=%u, token_limited_warps=%u, "
         "traversal_stack_limited_warps=%u, "
         "resident_rt_warps=%u\n",
         RTCORE_MAX_LANES_PER_WARP, RTCORE_HANDOFF_WINDOW_WORDS_PER_LANE,
         RTCORE_HANDOFF_WINDOW_BYTES_PER_LANE,
         RTCORE_HANDOFF_WINDOW_BYTES_PER_FULL_WARP,
         RTCORE_CONTEXT_BYTES_PER_LANE,
         profile.shared_memory_bytes_available_per_execution_partition,
         profile.max_resident_warps_per_execution_partition,
         profile.max_active_lanes_per_rt_warp,
         profile.handoff_window_bytes_per_active_rt_lane,
         profile.rt_tokens_per_execution_partition,
         profile.traversal_stack_entries_per_active_lane,
         profile.traversal_stack_entries_per_execution_partition,
         profile.traversal_stack_entries_per_full_warp,
         profile.handoff_window_limited_warps, profile.token_limited_warps,
         profile.traversal_stack_limited_warps,
         profile.resident_rt_warps);
  fflush(stdout);
}

bool rtcore_symbolic_submit_has_capacity(
    const ptx_instruction *pI, const rtcore_symbolic_resource_profile &profile,
    unsigned lane_slot_index, unsigned long long context_ptr,
    unsigned long long handoff_window_base) {
  const unsigned active_lane_mask = rtcore_active_thread_mask(pI);
  const unsigned active_lanes = rtcore_count_active_lanes(active_lane_mask);
  const bool resident_capacity_available = profile.resident_rt_warps != 0;

  printf("GPGPU-Sim PTX: RT_SUBMIT resource-check (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, active_lane_mask=0x%08x, active_lanes=%u, "
         "worst_case_divergent_lanes=%u, "
         "full_warp_handoff_window_bytes=%u, "
         "rt_tokens_per_execution_partition=%u, "
         "traversal_stack_entries_per_active_lane=%u, "
         "traversal_stack_entries_per_execution_partition=%u, "
         "traversal_stack_entries_per_full_warp=%u, "
         "traversal_stack_limited_warps=%u, "
         "resident_rt_warps=%u, capacity_available=%u\n",
         pI->source_file(), pI->source_line(), context_ptr,
         handoff_window_base, lane_slot_index, active_lane_mask, active_lanes,
         RTCORE_MAX_LANES_PER_WARP, profile.full_warp_handoff_window_bytes,
         profile.rt_tokens_per_execution_partition,
         profile.traversal_stack_entries_per_active_lane,
         profile.traversal_stack_entries_per_execution_partition,
         profile.traversal_stack_entries_per_full_warp,
         profile.traversal_stack_limited_warps,
         profile.resident_rt_warps, resident_capacity_available ? 1 : 0);
  fflush(stdout);

  if (profile.resident_rt_warps == 0) {
    printf("GPGPU-Sim PTX: RT_SUBMIT resource-wait-or-reject (%s:%u), "
           "reason=%s, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, "
           "traversal_stack_entries_per_execution_partition=%u, "
           "traversal_stack_entries_per_full_warp=%u, "
           "traversal_stack_limited_warps=%u, resident_rt_warps=%u, "
           "consumed=0\n",
           pI->source_file(), pI->source_line(),
           rtcore_symbolic_resource_limit_name(profile), context_ptr,
           handoff_window_base, lane_slot_index,
           profile.traversal_stack_entries_per_execution_partition,
           profile.traversal_stack_entries_per_full_warp,
           profile.traversal_stack_limited_warps, profile.resident_rt_warps);
    fflush(stdout);
  }

  return resident_capacity_available;
}

void rtcore_reject_symbolic_submit(const ptx_instruction *pI) {
  inst_not_implemented(pI);
}

bool rtcore_symbolic_submit_lane_is_active(
    const ptx_instruction *pI, unsigned lane_slot_index,
    unsigned long long context_ptr, unsigned long long handoff_window_base,
    unsigned active_lane_mask) {
  const unsigned lane_thread_mask =
      rtcore_lane_thread_mask(lane_slot_index);
  const bool test_active_mask_override =
      rtcore_test_inactive_lane_submit_enabled();
  if (test_active_mask_override) {
    active_lane_mask &= ~lane_thread_mask;
  }
  const bool lane_active =
      rtcore_lane_is_active_in_mask(active_lane_mask, lane_slot_index);

  printf("GPGPU-Sim PTX: RT_SUBMIT active-mask-check (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, test_active_mask_override=%u, "
         "active_lane_mask=0x%08x, lane_thread_mask=0x%08x, active=%u\n",
         pI->source_file(), pI->source_line(), context_ptr,
         handoff_window_base, lane_slot_index,
         test_active_mask_override ? 1 : 0, active_lane_mask, lane_thread_mask,
         lane_active ? 1 : 0);
  fflush(stdout);

  if (!lane_active) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=INACTIVE_LANE_SUBMIT, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, "
           "active_lane_mask=0x%08x, lane_thread_mask=0x%08x, consumed=0\n",
           pI->source_file(), pI->source_line(), context_ptr,
           handoff_window_base, lane_slot_index, active_lane_mask,
           lane_thread_mask);
    fflush(stdout);
  }

  return lane_active;
}

bool rtcore_synthetic_handoff_window_is_live(
    const rtcore_synthetic_handoff_key &key) {
  return g_rtcore_synthetic_handoff_windows.find(key) !=
         g_rtcore_synthetic_handoff_windows.end();
}

bool rtcore_symbolic_submit_lane_slot_available(
    const ptx_instruction *pI, const rtcore_synthetic_handoff_key &key,
    unsigned long long context_ptr) {
  const bool live_slot = rtcore_synthetic_handoff_window_is_live(key);
  const bool slot_available = !live_slot;

  printf("GPGPU-Sim PTX: RT_SUBMIT lane-slot-check (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, owner_hw_sid=%u, owner_hw_wid=%u, "
         "live_slot=%u, slot_available=%u\n",
         pI->source_file(), pI->source_line(), context_ptr,
         key.handoff_window_base, key.lane_slot_index, key.owner_hw_sid,
         key.owner_hw_wid, live_slot ? 1 : 0, slot_available ? 1 : 0);
  fflush(stdout);

  if (live_slot) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=LIVE_SLOT_OVERWRITE, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, "
           "owner_hw_sid=%u, owner_hw_wid=%u, consumed=0\n",
           pI->source_file(), pI->source_line(), context_ptr,
           key.handoff_window_base, key.lane_slot_index, key.owner_hw_sid,
           key.owner_hw_wid);
    fflush(stdout);
  }

  return slot_available;
}

rtcore_synthetic_handoff_key rtcore_make_synthetic_handoff_key(
    unsigned long long handoff_window_base, unsigned lane_slot_index,
    ptx_thread_info *thread) {
  rtcore_synthetic_handoff_key key;
  key.handoff_window_base = handoff_window_base;
  key.owner_hw_sid = thread->get_hw_sid();
  key.owner_hw_wid = thread->get_hw_wid();
  key.lane_slot_index = lane_slot_index;
  return key;
}

rtcore_symbolic_rt_token_key rtcore_make_symbolic_rt_token_key(
    unsigned long long context_ptr, unsigned long long handoff_window_base,
    unsigned lane_slot_index, ptx_thread_info *thread) {
  rtcore_symbolic_rt_token_key key;
  key.context_ptr = context_ptr;
  key.handoff_window_base = handoff_window_base;
  key.owner_hw_sid = thread->get_hw_sid();
  key.owner_hw_wid = thread->get_hw_wid();
  key.owner_hw_tid = thread->get_hw_tid();
  key.lane_slot_index = lane_slot_index;
  return key;
}

rtcore_symbolic_rt_token_reservation_key
rtcore_make_symbolic_rt_token_reservation_key(
    const ptx_thread_info::rtcore_current_warp_metadata &metadata,
    unsigned long long context_ptr, unsigned long long handoff_window_base) {
  rtcore_symbolic_rt_token_reservation_key key;
  key.warp_uid = metadata.warp_uid;
  key.warp_id = metadata.warp_id;
  key.owner_hw_sid = metadata.owner_hw_sid;
  key.static_inst_uid = metadata.static_inst_uid;
  key.active_mask = metadata.active_mask;
  key.context_ptr = context_ptr;
  key.handoff_window_base = handoff_window_base;
  return key;
}

bool rtcore_symbolic_rt_token_is_live(
    const rtcore_symbolic_rt_token_key &key) {
  return g_rtcore_symbolic_rt_tokens.find(key) !=
         g_rtcore_symbolic_rt_tokens.end();
}

bool rtcore_symbolic_submit_token_available(
    const ptx_instruction *pI, const rtcore_symbolic_rt_token_key &key) {
  const bool live_token = rtcore_symbolic_rt_token_is_live(key);
  const bool token_available = !live_token;

  printf("GPGPU-Sim PTX: RT_SUBMIT token-check (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, owner_hw_tid=%u, owner_hw_wid=%u, "
         "owner_hw_sid=%u, live_token=%u, token_available=%u\n",
         pI->source_file(), pI->source_line(), key.context_ptr,
         key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
         key.owner_hw_wid, key.owner_hw_sid, live_token ? 1 : 0,
         token_available ? 1 : 0);
  fflush(stdout);

  if (live_token) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=LIVE_TOKEN_REUSE, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, "
           "owner_hw_tid=%u, owner_hw_wid=%u, owner_hw_sid=%u, consumed=0\n",
           pI->source_file(), pI->source_line(), key.context_ptr,
           key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
           key.owner_hw_wid, key.owner_hw_sid);
    fflush(stdout);
  }

  return token_available;
}

size_t rtcore_symbolic_rt_token_allocator_live_count() {
  return g_rtcore_symbolic_rt_token_allocator.live_allocations;
}

size_t rtcore_symbolic_rt_token_allocator_free_count() {
  return g_rtcore_symbolic_rt_token_allocator.free_slot_ids.size();
}

size_t rtcore_symbolic_rt_token_allocator_slot_count() {
  return g_rtcore_symbolic_rt_token_allocator.slots.size();
}

size_t rtcore_symbolic_rt_token_allocator_live_plus_reserved_count(
    unsigned reserved_tokens) {
  return rtcore_symbolic_rt_token_allocator_live_count() + reserved_tokens;
}

bool rtcore_symbolic_rt_token_allocator_has_capacity(
    const rtcore_symbolic_resource_profile &profile,
    unsigned reserved_tokens) {
  return rtcore_symbolic_rt_token_allocator_live_plus_reserved_count(
             reserved_tokens) <= profile.rt_tokens_per_execution_partition;
}

bool rtcore_symbolic_rt_token_allocator_record_matches(
    const rtcore_symbolic_rt_token_record &record) {
  if (record.allocator_slot_id == 0 || record.allocator_generation == 0) {
    return false;
  }
  std::map<unsigned, rtcore_symbolic_rt_token_allocator_slot>::const_iterator
      slot = g_rtcore_symbolic_rt_token_allocator.slots.find(
          record.allocator_slot_id);
  return slot != g_rtcore_symbolic_rt_token_allocator.slots.end() &&
         slot->second.live &&
         slot->second.generation == record.allocator_generation;
}

rtcore_symbolic_rt_token_allocation
rtcore_allocate_symbolic_rt_token_slot(
    const ptx_instruction *pI, const rtcore_symbolic_rt_token_key &key,
    const rtcore_symbolic_resource_profile &profile) {
  rtcore_symbolic_rt_token_allocation allocation;
  allocation.slot_id = 0;
  allocation.generation = 0;
  allocation.allocated = false;
  bool reused_slot = false;
  rtcore_symbolic_rt_token_allocator_slot *slot = NULL;

  if (!g_rtcore_symbolic_rt_token_allocator.free_slot_ids.empty()) {
    allocation.slot_id =
        g_rtcore_symbolic_rt_token_allocator.free_slot_ids.back();
    g_rtcore_symbolic_rt_token_allocator.free_slot_ids.pop_back();
    std::map<unsigned, rtcore_symbolic_rt_token_allocator_slot>::iterator
        free_slot =
            g_rtcore_symbolic_rt_token_allocator.slots.find(
                allocation.slot_id);
    if (free_slot != g_rtcore_symbolic_rt_token_allocator.slots.end() &&
        !free_slot->second.live) {
      slot = &free_slot->second;
      reused_slot = true;
    }
  } else if (g_rtcore_symbolic_rt_token_allocator.slots.size() <
             profile.rt_tokens_per_execution_partition) {
    rtcore_symbolic_rt_token_allocator_slot new_slot;
    allocation.slot_id =
        g_rtcore_symbolic_rt_token_allocator.next_slot_id++;
    allocation.generation =
        g_rtcore_symbolic_rt_token_allocator.next_generation++;
    new_slot.slot_id = allocation.slot_id;
    new_slot.generation = allocation.generation;
    new_slot.live = false;
    std::pair<std::map<unsigned, rtcore_symbolic_rt_token_allocator_slot>::
                  iterator,
              bool>
        inserted = g_rtcore_symbolic_rt_token_allocator.slots.insert(
            std::make_pair(new_slot.slot_id, new_slot));
    if (inserted.second) {
      slot = &inserted.first->second;
    }
  }

  if (slot != NULL) {
    slot->live = true;
    allocation.slot_id = slot->slot_id;
    allocation.generation = slot->generation;
    allocation.allocated = true;
    g_rtcore_symbolic_rt_token_allocator.live_allocations++;
  }

  printf("GPGPU-Sim PTX: RT_SUBMIT allocator-allocate (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, owner_hw_tid=%u, owner_hw_wid=%u, "
         "owner_hw_sid=%u, allocator_slot_id=%u, "
         "allocator_generation=%u, allocator_live=%zu, "
         "allocator_released=%zu, allocator_free=%zu, "
         "allocator_slots=%zu, allocator_slot_reused=%u, "
         "rt_tokens_per_execution_partition=%u, allocated=%u\n",
         pI->source_file(), pI->source_line(), key.context_ptr,
         key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
         key.owner_hw_wid, key.owner_hw_sid, allocation.slot_id,
         allocation.generation,
         g_rtcore_symbolic_rt_token_allocator.live_allocations,
         g_rtcore_symbolic_rt_token_allocator.released_allocations,
         rtcore_symbolic_rt_token_allocator_free_count(),
         rtcore_symbolic_rt_token_allocator_slot_count(),
         reused_slot ? 1 : 0, profile.rt_tokens_per_execution_partition,
         allocation.allocated ? 1 : 0);
  fflush(stdout);

  return allocation;
}

bool rtcore_release_symbolic_rt_token_allocation(
    const rtcore_symbolic_rt_token_record &record) {
  const bool valid_metadata =
      record.allocator_slot_id != 0 && record.allocator_generation != 0;
  const bool valid_allocation =
      valid_metadata &&
      rtcore_symbolic_rt_token_allocator_record_matches(record);
  if (!valid_allocation ||
      g_rtcore_symbolic_rt_token_allocator.live_allocations == 0) {
    return false;
  }
  std::map<unsigned, rtcore_symbolic_rt_token_allocator_slot>::iterator slot =
      g_rtcore_symbolic_rt_token_allocator.slots.find(
          record.allocator_slot_id);
  slot->second.live = false;
  slot->second.generation =
      g_rtcore_symbolic_rt_token_allocator.next_generation++;
  g_rtcore_symbolic_rt_token_allocator.free_slot_ids.push_back(
      record.allocator_slot_id);
  g_rtcore_symbolic_rt_token_allocator.live_allocations--;
  g_rtcore_symbolic_rt_token_allocator.released_allocations++;
  return true;
}

bool rtcore_acquire_symbolic_rt_token(
    const ptx_instruction *pI, const rtcore_symbolic_rt_token_key &key,
    unsigned window_generation, unsigned completion_seq, unsigned resume_seq,
    unsigned window_tag, unsigned result_word) {
  const rtcore_symbolic_resource_profile profile =
      rtcore_get_symbolic_resource_profile();
  const rtcore_symbolic_rt_token_allocation allocation =
      rtcore_allocate_symbolic_rt_token_slot(pI, key, profile);
  if (!allocation.allocated) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=RT_TOKEN_ALLOCATOR_EXHAUSTED_AT_ACQUIRE, "
           "context_ptr=0x%llx, handoff_window_base=0x%llx, "
           "lane_slot_index=%u, rt_tokens_per_execution_partition=%u, "
           "allocator_live=%zu, allocator_free=%zu, allocator_slots=%zu, "
           "consumed=0\n",
           pI->source_file(), pI->source_line(), key.context_ptr,
           key.handoff_window_base, key.lane_slot_index,
           profile.rt_tokens_per_execution_partition,
           rtcore_symbolic_rt_token_allocator_live_count(),
           rtcore_symbolic_rt_token_allocator_free_count(),
           rtcore_symbolic_rt_token_allocator_slot_count());
    fflush(stdout);
    return false;
  }
  rtcore_symbolic_rt_token_record record;
  record.token_id = g_rtcore_next_symbolic_rt_token_id++;
  record.allocator_slot_id = allocation.slot_id;
  record.allocator_generation = allocation.generation;
  record.window_generation = window_generation;
  record.completion_seq = completion_seq;
  record.resume_seq = resume_seq;
  record.window_tag = window_tag;
  record.result_word = result_word;
  record.completed = false;

  const bool inserted =
      g_rtcore_symbolic_rt_tokens.insert(std::make_pair(key, record)).second;
  if (!inserted) {
    rtcore_release_symbolic_rt_token_allocation(record);
  }

  printf("GPGPU-Sim PTX: RT_SUBMIT token-acquire (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, owner_hw_tid=%u, owner_hw_wid=%u, "
         "owner_hw_sid=%u, token_id=%u, allocator_slot_id=%u, "
         "allocator_generation=%u, window_generation=%u, "
         "completion_seq=%u, resume_seq=%u, window_tag=%u, result=0x%08x, "
         "acquired=%u, live_tokens=%zu, allocator_live=%zu, "
         "allocator_free=%zu, allocator_slots=%zu\n",
         pI->source_file(), pI->source_line(), key.context_ptr,
         key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
         key.owner_hw_wid, key.owner_hw_sid, record.token_id,
         record.allocator_slot_id, record.allocator_generation,
         record.window_generation, record.completion_seq, record.resume_seq,
         record.window_tag, record.result_word, inserted ? 1 : 0,
         g_rtcore_symbolic_rt_tokens.size(),
         rtcore_symbolic_rt_token_allocator_live_count(),
         rtcore_symbolic_rt_token_allocator_free_count(),
         rtcore_symbolic_rt_token_allocator_slot_count());
  fflush(stdout);

  if (!inserted) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=LIVE_TOKEN_REUSE_AT_ACQUIRE, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, consumed=0\n",
           pI->source_file(), pI->source_line(), key.context_ptr,
           key.handoff_window_base, key.lane_slot_index);
    fflush(stdout);
  }

  return inserted;
}

bool rtcore_complete_symbolic_rt_token(
    const ptx_instruction *pI, const rtcore_symbolic_rt_token_key &key,
    unsigned result_word, unsigned completion_seq, unsigned resume_seq,
    unsigned window_tag) {
  std::map<rtcore_symbolic_rt_token_key,
           rtcore_symbolic_rt_token_record>::iterator token =
      g_rtcore_symbolic_rt_tokens.find(key);
  const bool live_token = token != g_rtcore_symbolic_rt_tokens.end();
  const unsigned allocator_slot_id =
      live_token ? token->second.allocator_slot_id : 0;
  const unsigned allocator_generation =
      live_token ? token->second.allocator_generation : 0;
  const bool allocator_record_matches =
      live_token &&
      rtcore_symbolic_rt_token_allocator_record_matches(token->second);
  const bool allocator_metadata_valid =
      allocator_record_matches;
  const bool result_matches = live_token && token->second.result_word == result_word;
  const bool completion_seq_matches =
      live_token && token->second.completion_seq == completion_seq;
  const bool resume_seq_matches =
      live_token && token->second.resume_seq == resume_seq;
  const bool window_tag_matches =
      live_token && token->second.window_tag == window_tag;
  const bool completion_validated =
      live_token && allocator_metadata_valid && result_matches &&
      completion_seq_matches && resume_seq_matches && window_tag_matches;
  const bool test_retire_before_completion =
      completion_validated && rtcore_test_retire_before_completion_enabled();

  if (completion_validated && !test_retire_before_completion) {
    token->second.completed = true;
  }
  const bool completed = live_token && token->second.completed;

  printf("GPGPU-Sim PTX: RT_SUBMIT token-complete (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, owner_hw_tid=%u, owner_hw_wid=%u, "
         "owner_hw_sid=%u, allocator_slot_id=%u, "
         "allocator_generation=%u, live_token=%u, "
         "allocator_metadata_valid=%u, result_match=%u, "
         "completion_seq_match=%u, resume_seq_match=%u, "
         "window_tag_match=%u, test_retire_before_completion=%u, "
         "completed=%u\n",
         pI->source_file(), pI->source_line(), key.context_ptr,
         key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
         key.owner_hw_wid, key.owner_hw_sid, allocator_slot_id,
         allocator_generation, live_token ? 1 : 0,
         allocator_metadata_valid ? 1 : 0, result_matches ? 1 : 0,
         completion_seq_matches ? 1 : 0,
         resume_seq_matches ? 1 : 0, window_tag_matches ? 1 : 0,
         test_retire_before_completion ? 1 : 0,
         completed ? 1 : 0);
  fflush(stdout);

  return completion_validated;
}

size_t rtcore_synthetic_handoff_window_count();
size_t rtcore_symbolic_rt_token_count();
size_t rtcore_symbolic_rt_token_reservation_count();

bool rtcore_symbolic_rt_token_can_retire(
    const ptx_instruction *pI, const rtcore_symbolic_rt_token_key &key) {
  std::map<rtcore_symbolic_rt_token_key,
           rtcore_symbolic_rt_token_record>::const_iterator token =
      g_rtcore_symbolic_rt_tokens.find(key);
  const bool live_token = token != g_rtcore_symbolic_rt_tokens.end();
  const bool completed = live_token && token->second.completed;
  const bool allocator_record_matches =
      live_token &&
      rtcore_symbolic_rt_token_allocator_record_matches(token->second);
  const bool test_stale_allocator_generation =
      live_token && completed &&
      rtcore_test_stale_allocator_generation_enabled();
  const bool allocator_generation_matches =
      allocator_record_matches && !test_stale_allocator_generation;
  const bool accepted = live_token && completed && allocator_generation_matches;

  printf("GPGPU-Sim PTX: RT_RETIRE_CONTEXT token-check (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, owner_hw_tid=%u, owner_hw_wid=%u, "
         "owner_hw_sid=%u, live_token=%u, completed=%u, "
         "allocator_generation_matches=%u, "
         "test_stale_allocator_generation=%u, accepted=%u\n",
         pI->source_file(), pI->source_line(), key.context_ptr,
         key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
         key.owner_hw_wid, key.owner_hw_sid, live_token ? 1 : 0,
         completed ? 1 : 0, allocator_generation_matches ? 1 : 0,
         test_stale_allocator_generation ? 1 : 0,
         accepted ? 1 : 0);
  fflush(stdout);

  if (!live_token) {
    printf("GPGPU-Sim PTX: RT_RETIRE_CONTEXT fail-closed (%s:%u), "
           "reason=TOKEN_NOT_LIVE, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, "
           "owner_hw_tid=%u, owner_hw_wid=%u, owner_hw_sid=%u\n",
           pI->source_file(), pI->source_line(), key.context_ptr,
           key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
           key.owner_hw_wid, key.owner_hw_sid);
    fflush(stdout);
  } else if (!completed) {
    printf("GPGPU-Sim PTX: RT_RETIRE_CONTEXT fail-closed (%s:%u), "
           "reason=RETIRE_BEFORE_COMPLETION, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, "
           "lane_slot_index=%u, owner_hw_tid=%u, owner_hw_wid=%u, "
           "owner_hw_sid=%u\n",
           pI->source_file(), pI->source_line(), key.context_ptr,
           key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
           key.owner_hw_wid, key.owner_hw_sid);
    fflush(stdout);
  } else if (!allocator_generation_matches) {
    printf("GPGPU-Sim PTX: RT_RETIRE_CONTEXT fail-closed (%s:%u), "
           "reason=RT_TOKEN_ALLOCATOR_STALE_GENERATION, "
           "context_ptr=0x%llx, handoff_window_base=0x%llx, "
           "lane_slot_index=%u, owner_hw_tid=%u, owner_hw_wid=%u, "
           "owner_hw_sid=%u, remaining_windows=%zu, remaining_tokens=%zu, "
           "remaining_reservations=%zu, remaining_allocator_live=%zu\n",
           pI->source_file(), pI->source_line(), key.context_ptr,
           key.handoff_window_base, key.lane_slot_index, key.owner_hw_tid,
           key.owner_hw_wid, key.owner_hw_sid,
           rtcore_synthetic_handoff_window_count(),
           rtcore_symbolic_rt_token_count(),
           rtcore_symbolic_rt_token_reservation_count(),
           rtcore_symbolic_rt_token_allocator_live_count());
    fflush(stdout);
  }

  return accepted;
}

bool rtcore_release_symbolic_rt_token(
    const rtcore_symbolic_rt_token_key &key) {
  std::map<rtcore_symbolic_rt_token_key,
           rtcore_symbolic_rt_token_record>::iterator token =
      g_rtcore_symbolic_rt_tokens.find(key);
  if (token == g_rtcore_symbolic_rt_tokens.end()) {
    return false;
  }
  const bool released_allocation =
      rtcore_release_symbolic_rt_token_allocation(token->second);
  if (!released_allocation) {
    return false;
  }
  g_rtcore_symbolic_rt_tokens.erase(token);
  return released_allocation;
}

size_t rtcore_symbolic_rt_token_count() {
  return g_rtcore_symbolic_rt_tokens.size();
}

bool rtcore_cancel_symbolic_rt_token_reservation(
    const rtcore_symbolic_rt_token_reservation_key &key) {
  return g_rtcore_symbolic_rt_token_reservations.erase(key) == 1;
}

size_t rtcore_symbolic_rt_token_reservation_count() {
  return g_rtcore_symbolic_rt_token_reservations.size();
}

size_t rtcore_synthetic_handoff_window_count();

bool rtcore_symbolic_submit_token_reservation_available(
    const ptx_instruction *pI, const rtcore_symbolic_resource_profile &profile,
    const rtcore_symbolic_rt_token_reservation_key &key,
    const rtcore_symbolic_rt_token_key &token_key, unsigned lane_slot_index) {
  const size_t token_map_live_tokens = rtcore_symbolic_rt_token_count();
  const unsigned active_lanes = rtcore_count_active_lanes(key.active_mask);
  const unsigned reserved_tokens = active_lanes;
  const size_t allocator_live_tokens =
      rtcore_symbolic_rt_token_allocator_live_count();
  const size_t allocator_live_plus_reserved_tokens =
      rtcore_symbolic_rt_token_allocator_live_plus_reserved_count(
          reserved_tokens);
  const bool allocator_capacity_available =
      rtcore_symbolic_rt_token_allocator_has_capacity(
          profile, reserved_tokens);
  const size_t live_tokens = allocator_live_tokens;
  std::map<rtcore_symbolic_rt_token_reservation_key,
           rtcore_symbolic_rt_token_reservation_record>::iterator reservation =
      g_rtcore_symbolic_rt_token_reservations.find(key);
  const bool reservation_live =
      reservation != g_rtcore_symbolic_rt_token_reservations.end();
  bool reservation_inserted = false;
  bool token_capacity_available = reservation_live;

  if (!reservation_live) {
    token_capacity_available = allocator_capacity_available;
    if (token_capacity_available) {
      rtcore_symbolic_rt_token_reservation_record record;
      record.active_mask = key.active_mask;
      record.active_lanes = active_lanes;
      record.acquired_lane_mask = 0;
      reservation_inserted =
          g_rtcore_symbolic_rt_token_reservations.insert(
              std::make_pair(key, record))
              .second;
      token_capacity_available = reservation_inserted;
    }
  }

  printf("GPGPU-Sim PTX: RT_SUBMIT token-reservation-check (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, owner_hw_tid=%u, owner_hw_wid=%u, "
         "owner_hw_sid=%u, warp_uid=%u, static_inst_uid=%u, "
         "active_lane_mask=0x%08x, active_lanes=%u, "
         "reservation_live=%u, reservation_inserted=%u, "
         "reserved_tokens=%u, rt_tokens_per_execution_partition=%u, "
         "live_tokens=%zu, token_map_live_tokens=%zu, "
         "allocator_live_tokens=%zu, allocator_reserved_tokens=%u, "
         "allocator_live_plus_reserved_tokens=%zu, "
         "token_capacity_available=%u\n",
         pI->source_file(), pI->source_line(), key.context_ptr,
         key.handoff_window_base, lane_slot_index, token_key.owner_hw_tid,
         key.warp_id, key.owner_hw_sid, key.warp_uid, key.static_inst_uid,
         key.active_mask, active_lanes, reservation_live ? 1 : 0,
         reservation_inserted ? 1 : 0, reserved_tokens,
         profile.rt_tokens_per_execution_partition, live_tokens,
         token_map_live_tokens, allocator_live_tokens, reserved_tokens,
         allocator_live_plus_reserved_tokens,
         token_capacity_available ? 1 : 0);
  fflush(stdout);

  if (!token_capacity_available) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=RT_TOKEN_CAPACITY_EXHAUSTED, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, "
           "owner_hw_tid=%u, owner_hw_wid=%u, owner_hw_sid=%u, "
           "warp_uid=%u, static_inst_uid=%u, active_lane_mask=0x%08x, "
           "active_lanes=%u, reservation_live=%u, reserved_tokens=%u, "
           "rt_tokens_per_execution_partition=%u, live_tokens=%zu, "
           "token_map_live_tokens=%zu, allocator_live_tokens=%zu, "
           "allocator_reserved_tokens=%u, "
           "allocator_live_plus_reserved_tokens=%zu, "
           "token_capacity_available=%u, remaining_windows=%zu, "
           "remaining_tokens=%zu, remaining_reservations=%zu, "
           "remaining_allocator_live=%zu, consumed=0\n",
           pI->source_file(), pI->source_line(), key.context_ptr,
           key.handoff_window_base, lane_slot_index, token_key.owner_hw_tid,
           key.warp_id, key.owner_hw_sid, key.warp_uid, key.static_inst_uid,
           key.active_mask, active_lanes, reservation_live ? 1 : 0,
           reserved_tokens, profile.rt_tokens_per_execution_partition,
           live_tokens, token_map_live_tokens, allocator_live_tokens,
           reserved_tokens, allocator_live_plus_reserved_tokens,
           token_capacity_available ? 1 : 0,
           rtcore_synthetic_handoff_window_count(),
           rtcore_symbolic_rt_token_count(),
           rtcore_symbolic_rt_token_reservation_count(),
           rtcore_symbolic_rt_token_allocator_live_count());
    fflush(stdout);
  }

  return token_capacity_available;
}

bool rtcore_note_symbolic_rt_token_reservation_lane_acquired(
    const ptx_instruction *pI,
    const rtcore_symbolic_rt_token_reservation_key &key,
    unsigned lane_slot_index) {
  std::map<rtcore_symbolic_rt_token_reservation_key,
           rtcore_symbolic_rt_token_reservation_record>::iterator reservation =
      g_rtcore_symbolic_rt_token_reservations.find(key);
  const bool reservation_live =
      reservation != g_rtcore_symbolic_rt_token_reservations.end();
  const unsigned lane_thread_mask = rtcore_lane_thread_mask(lane_slot_index);
  unsigned acquired_lane_mask = 0;
  unsigned active_lanes = 0;
  bool reservation_complete = false;

  if (reservation_live) {
    reservation->second.acquired_lane_mask |= lane_thread_mask;
    acquired_lane_mask = reservation->second.acquired_lane_mask;
    active_lanes = reservation->second.active_lanes;
    reservation_complete =
        (acquired_lane_mask & reservation->second.active_mask) ==
        reservation->second.active_mask;
    if (reservation_complete) {
      g_rtcore_symbolic_rt_token_reservations.erase(reservation);
    }
  }

  printf("GPGPU-Sim PTX: RT_SUBMIT token-reservation-acquire (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, warp_uid=%u, static_inst_uid=%u, "
         "active_lane_mask=0x%08x, active_lanes=%u, "
         "lane_thread_mask=0x%08x, acquired_lane_mask=0x%08x, "
         "reservation_live=%u, reservation_complete=%u\n",
         pI->source_file(), pI->source_line(), key.context_ptr,
         key.handoff_window_base, lane_slot_index, key.warp_uid,
         key.static_inst_uid, key.active_mask, active_lanes,
         lane_thread_mask, acquired_lane_mask, reservation_live ? 1 : 0,
         reservation_complete ? 1 : 0);
  fflush(stdout);

  if (!reservation_live) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=RT_TOKEN_RESERVATION_MISSING, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, "
           "warp_uid=%u, static_inst_uid=%u, consumed=0\n",
           pI->source_file(), pI->source_line(), key.context_ptr,
           key.handoff_window_base, lane_slot_index, key.warp_uid,
           key.static_inst_uid);
    fflush(stdout);
  }

  return reservation_live;
}

void rtcore_populate_synthetic_owner_tuple(
    rtcore_synthetic_handoff_header *header,
    const rtcore_synthetic_handoff_key &key, ptx_thread_info *thread,
    unsigned thread_mask) {
  header->handoff_window_base = key.handoff_window_base;
  header->lane_slot_index = key.lane_slot_index;
  header->lane_slot_base = rtcore_handoff_lane_slot_base(
      key.handoff_window_base, key.lane_slot_index);
  header->owner_hw_tid = thread->get_hw_tid();
  header->owner_hw_wid = thread->get_hw_wid();
  header->owner_hw_sid = thread->get_hw_sid();
  header->thread_mask = thread_mask;
  header->window_state = RTCORE_WINDOW_STATE_COMPLETE;
}

bool rtcore_synthetic_owner_tuple_matches(
    const rtcore_synthetic_handoff_header &window,
    const rtcore_synthetic_handoff_key &key, unsigned long long context_ptr,
    const ptx_instruction *pI, ptx_thread_info *thread) {
  const unsigned lane_thread_mask =
      rtcore_lane_thread_mask(key.lane_slot_index);
  return window.context_ptr == context_ptr &&
         window.handoff_window_base == key.handoff_window_base &&
         window.lane_slot_index == key.lane_slot_index &&
         window.owner_hw_sid == key.owner_hw_sid &&
         window.owner_hw_wid == key.owner_hw_wid &&
         window.lane_slot_base == rtcore_handoff_lane_slot_base(
                                      key.handoff_window_base,
                                      key.lane_slot_index) &&
         window.owner_hw_tid == thread->get_hw_tid() &&
         window.owner_hw_wid == thread->get_hw_wid() &&
         window.owner_hw_sid == thread->get_hw_sid() &&
         (window.thread_mask & lane_thread_mask) != 0 &&
         window.window_state == RTCORE_WINDOW_STATE_COMPLETE;
}

void rtcore_publish_synthetic_handoff_window(
    const ptx_instruction *pI, const rtcore_synthetic_handoff_key &key,
    const rtcore_synthetic_handoff_header &header) {
  const bool inserted =
      g_rtcore_synthetic_handoff_windows.insert(std::make_pair(key, header))
          .second;
  if (!inserted) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=LIVE_SLOT_OVERWRITE_AT_PUBLISH, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u\n",
           pI->source_file(), pI->source_line(), header.context_ptr,
           key.handoff_window_base, key.lane_slot_index);
    fflush(stdout);
  }
  assert(inserted);
  rtcore_commit_synthetic_window_generation(key, header.window_generation);
  const unsigned lane_slot_byte_offset =
      rtcore_handoff_lane_slot_byte_offset(key.lane_slot_index);
  const unsigned long long lane_slot_base =
      rtcore_handoff_lane_slot_base(key.handoff_window_base,
                                    key.lane_slot_index);

  printf("GPGPU-Sim PTX: RT_SUBMIT handoff-window-published (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, lane_slot_byte_offset=%u, "
         "lane_slot_base=0x%llx, owner_hw_tid=%u, owner_hw_wid=%u, "
         "owner_hw_sid=%u, thread_mask=0x%08x, window_generation=%u, "
         "completion_seq=%u, resume_seq=%u, window_tag=%u, w0=0x%08x, "
         "w1=0x%08x, w2=0x%08x, w3=0x%08x\n",
         pI->source_file(), pI->source_line(), header.context_ptr,
         key.handoff_window_base, key.lane_slot_index, lane_slot_byte_offset,
         lane_slot_base, header.owner_hw_tid, header.owner_hw_wid,
         header.owner_hw_sid, header.thread_mask, header.window_generation,
         header.completion_seq, header.resume_seq, header.window_tag,
         header.w0, header.w1, header.w2, header.w3);
  fflush(stdout);
}

bool rtcore_synthetic_result_lane_binding_matches(
    const ptx_instruction *pI, const rtcore_synthetic_handoff_key &key,
    const rtcore_synthetic_handoff_header &header,
    unsigned long long context_ptr, unsigned result_word) {
  const bool context_matches = header.context_ptr == context_ptr;
  const bool window_base_matches =
      header.handoff_window_base == key.handoff_window_base;
  const bool lane_slot_matches =
      header.lane_slot_index == key.lane_slot_index;
  const bool lane_slot_base_matches =
      header.lane_slot_base == rtcore_handoff_lane_slot_base(
                                   key.handoff_window_base,
                                   key.lane_slot_index);
  const bool result_matches_header = header.w0 == result_word;
  const unsigned lane_thread_mask =
      rtcore_lane_thread_mask(key.lane_slot_index);
  const bool active_lane = (header.thread_mask & lane_thread_mask) != 0;
  const bool accepted =
      context_matches && window_base_matches && lane_slot_matches &&
      lane_slot_base_matches && result_matches_header && active_lane;

  printf("GPGPU-Sim PTX: RT_SUBMIT result-lane-binding (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, lane_slot_base=0x%llx, result=0x%08x, "
         "context_match=%u, window_base_match=%u, lane_slot_match=%u, "
         "lane_slot_base_match=%u, result_match=%u, active_lane=%u, "
         "accepted=%u\n",
         pI->source_file(), pI->source_line(), context_ptr,
         key.handoff_window_base, key.lane_slot_index,
         rtcore_handoff_lane_slot_base(key.handoff_window_base,
                                       key.lane_slot_index),
         result_word, context_matches ? 1 : 0,
         window_base_matches ? 1 : 0, lane_slot_matches ? 1 : 0,
         lane_slot_base_matches ? 1 : 0, result_matches_header ? 1 : 0,
         active_lane ? 1 : 0, accepted ? 1 : 0);
  fflush(stdout);

  return accepted;
}

const rtcore_synthetic_handoff_header *rtcore_acquire_synthetic_handoff_window(
    const std::map<rtcore_synthetic_handoff_key, rtcore_synthetic_handoff_header>
        &windows,
    const rtcore_synthetic_handoff_key &key) {
  std::map<rtcore_synthetic_handoff_key,
           rtcore_synthetic_handoff_header>::const_iterator window =
      windows.find(key);
  if (window == windows.end()) {
    return NULL;
  }
  return &window->second;
}

bool rtcore_release_synthetic_handoff_window(
    const rtcore_synthetic_handoff_key &key) {
  return g_rtcore_synthetic_handoff_windows.erase(key) == 1;
}

size_t rtcore_synthetic_handoff_window_count() {
  return g_rtcore_synthetic_handoff_windows.size();
}

void rtcore_rollback_symbolic_submit_after_token_acquire(
    const ptx_instruction *pI,
    const rtcore_symbolic_rt_token_key &token_key,
    const rtcore_symbolic_rt_token_reservation_key &reservation_key,
    const rtcore_synthetic_handoff_key &handoff_key) {
  const bool released_token =
      rtcore_release_symbolic_rt_token(token_key);
  const bool cancelled_reservation =
      rtcore_cancel_symbolic_rt_token_reservation(reservation_key);

  printf("GPGPU-Sim PTX: RT_SUBMIT rollback-cleanup (%s:%u), "
         "reason=FAIL_AFTER_TOKEN_ACQUIRE, context_ptr=0x%llx, "
         "handoff_window_base=0x%llx, lane_slot_index=%u, "
         "owner_hw_tid=%u, owner_hw_wid=%u, owner_hw_sid=%u, "
         "warp_uid=%u, static_inst_uid=%u, released_token=%u, "
         "cancelled_reservation=%u, remaining_windows=%zu, "
         "remaining_tokens=%zu, remaining_reservations=%zu, "
         "remaining_allocator_live=%zu\n",
         pI->source_file(), pI->source_line(), token_key.context_ptr,
         handoff_key.handoff_window_base, handoff_key.lane_slot_index,
         token_key.owner_hw_tid, token_key.owner_hw_wid,
         token_key.owner_hw_sid, reservation_key.warp_uid,
         reservation_key.static_inst_uid, released_token ? 1 : 0,
         cancelled_reservation ? 1 : 0,
         rtcore_synthetic_handoff_window_count(),
         rtcore_symbolic_rt_token_count(),
         rtcore_symbolic_rt_token_reservation_count(),
         rtcore_symbolic_rt_token_allocator_live_count());
  fflush(stdout);

  printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
         "reason=FAIL_AFTER_TOKEN_ACQUIRE, context_ptr=0x%llx, "
         "handoff_window_base=0x%llx, lane_slot_index=%u, consumed=0\n",
         pI->source_file(), pI->source_line(), token_key.context_ptr,
         handoff_key.handoff_window_base, handoff_key.lane_slot_index);
  fflush(stdout);
}

void rtcore_rollback_symbolic_submit_after_handoff_publish(
    const ptx_instruction *pI,
    const rtcore_symbolic_rt_token_key &token_key,
    const rtcore_symbolic_rt_token_reservation_key &reservation_key,
    const rtcore_synthetic_handoff_key &handoff_key) {
  const bool released_window =
      rtcore_release_synthetic_handoff_window(handoff_key);
  const bool released_token =
      rtcore_release_symbolic_rt_token(token_key);
  const bool cancelled_reservation =
      rtcore_cancel_symbolic_rt_token_reservation(reservation_key);

  printf("GPGPU-Sim PTX: RT_SUBMIT rollback-cleanup (%s:%u), "
         "reason=FAIL_AFTER_HANDOFF_PUBLISH, context_ptr=0x%llx, "
         "handoff_window_base=0x%llx, lane_slot_index=%u, "
         "owner_hw_tid=%u, owner_hw_wid=%u, owner_hw_sid=%u, "
         "warp_uid=%u, static_inst_uid=%u, released_window=%u, "
         "released_token=%u, cancelled_reservation=%u, "
         "remaining_windows=%zu, remaining_tokens=%zu, "
         "remaining_reservations=%zu, remaining_allocator_live=%zu\n",
         pI->source_file(), pI->source_line(), token_key.context_ptr,
         handoff_key.handoff_window_base, handoff_key.lane_slot_index,
         token_key.owner_hw_tid, token_key.owner_hw_wid,
         token_key.owner_hw_sid, reservation_key.warp_uid,
         reservation_key.static_inst_uid, released_window ? 1 : 0,
         released_token ? 1 : 0, cancelled_reservation ? 1 : 0,
         rtcore_synthetic_handoff_window_count(),
         rtcore_symbolic_rt_token_count(),
         rtcore_symbolic_rt_token_reservation_count(),
         rtcore_symbolic_rt_token_allocator_live_count());
  fflush(stdout);

  printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
         "reason=FAIL_AFTER_HANDOFF_PUBLISH, context_ptr=0x%llx, "
         "handoff_window_base=0x%llx, lane_slot_index=%u, consumed=0\n",
         pI->source_file(), pI->source_line(), token_key.context_ptr,
         handoff_key.handoff_window_base, handoff_key.lane_slot_index);
  fflush(stdout);
}

void rtcore_rollback_symbolic_submit_after_result_write(
    const ptx_instruction *pI,
    const rtcore_symbolic_rt_token_key &token_key,
    const rtcore_symbolic_rt_token_reservation_key &reservation_key,
    const rtcore_synthetic_handoff_key &handoff_key) {
  const bool released_window =
      rtcore_release_synthetic_handoff_window(handoff_key);
  const bool released_token =
      rtcore_release_symbolic_rt_token(token_key);
  const bool cancelled_reservation =
      rtcore_cancel_symbolic_rt_token_reservation(reservation_key);

  printf("GPGPU-Sim PTX: RT_SUBMIT rollback-cleanup (%s:%u), "
         "reason=FAIL_AFTER_RESULT_WRITE, context_ptr=0x%llx, "
         "handoff_window_base=0x%llx, lane_slot_index=%u, "
         "owner_hw_tid=%u, owner_hw_wid=%u, owner_hw_sid=%u, "
         "warp_uid=%u, static_inst_uid=%u, result_written=1, "
         "released_window=%u, released_token=%u, "
         "cancelled_reservation=%u, remaining_windows=%zu, "
         "remaining_tokens=%zu, remaining_reservations=%zu, "
         "remaining_allocator_live=%zu\n",
         pI->source_file(), pI->source_line(), token_key.context_ptr,
         handoff_key.handoff_window_base, handoff_key.lane_slot_index,
         token_key.owner_hw_tid, token_key.owner_hw_wid,
         token_key.owner_hw_sid, reservation_key.warp_uid,
         reservation_key.static_inst_uid, released_window ? 1 : 0,
         released_token ? 1 : 0, cancelled_reservation ? 1 : 0,
         rtcore_synthetic_handoff_window_count(),
         rtcore_symbolic_rt_token_count(),
         rtcore_symbolic_rt_token_reservation_count(),
         rtcore_symbolic_rt_token_allocator_live_count());
  fflush(stdout);

  printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
         "reason=FAIL_AFTER_RESULT_WRITE, context_ptr=0x%llx, "
         "handoff_window_base=0x%llx, lane_slot_index=%u, consumed=0\n",
         pI->source_file(), pI->source_line(), token_key.context_ptr,
         handoff_key.handoff_window_base, handoff_key.lane_slot_index);
  fflush(stdout);
}

unsigned rtcore_compact_result(unsigned reason, unsigned flags,
                               unsigned completion_seq_low,
                               unsigned resume_seq_low,
                               unsigned window_tag_low) {
  return RTCORE_COMPLETION_VALID | (reason & 0xf) | ((flags & 0xff) << 4) |
         ((completion_seq_low & 0xff) << 12) |
         ((resume_seq_low & 0xff) << 20) |
         ((window_tag_low & 0x7) << 28);
}

unsigned rtcore_float_to_u32(float value) {
  unsigned word = 0;
  memcpy(&word, &value, sizeof(word));
  return word;
}

const char *rtcore_return_reason_name(unsigned reason);

void rtcore_publish_synthetic_dependent_groups(
    const ptx_instruction *pI, const Traversal_data &traversal_data,
    unsigned reason, unsigned completion_seq_low, unsigned resume_seq_low,
    unsigned window_tag_low, rtcore_synthetic_handoff_header *header) {
  const bool closest_hit = reason == RTCORE_RETURN_FOR_CLOSEST_HIT;
  const bool memory_fault = reason == RTCORE_RETURN_FOR_MEMORY_FAULT;
  const unsigned dispatch_selector =
      closest_hit ? (unsigned)traversal_data.closest_hit.hitGroupIndex
                  : traversal_data.missIndex;

  if (!memory_fault) {
    header->w4 = rtcore_compact_result(
        reason, RTCORE_COMPLETION_FLAG_TRACE_DONE, completion_seq_low,
        resume_seq_low, window_tag_low);
    header->w5 = dispatch_selector;
    header->w6 =
        closest_hit ? (unsigned)traversal_data.closest_hit.geometryType : 0;
    header->w7 = 0;
  } else {
    header->w4 = 0;
    header->w5 = 0;
    header->w6 = 0;
    header->w7 = 0;
  }

  if (closest_hit && traversal_data.hit_geometry) {
    header->w8 = rtcore_compact_result(
        reason, RTCORE_COMPLETION_FLAG_TRACE_DONE, completion_seq_low,
        resume_seq_low, window_tag_low);
    header->w9 = (unsigned)traversal_data.closest_hit.geometryType;
    header->w10 = (unsigned)traversal_data.closest_hit.hitGroupIndex;
    header->w11 = traversal_data.closest_hit.primitive_index;
    header->w12 = traversal_data.closest_hit.instance_index;
    header->w13 =
        rtcore_float_to_u32(traversal_data.closest_hit.world_min_thit);
    header->w14 =
        rtcore_float_to_u32(traversal_data.closest_hit.barycentric_coordinates.x);
    header->w15 =
        rtcore_float_to_u32(traversal_data.closest_hit.barycentric_coordinates.y);
  } else {
    header->w8 = 0;
    header->w9 = 0;
    header->w10 = 0;
    header->w11 = 0;
    header->w12 = 0;
    header->w13 = 0;
    header->w14 = 0;
    header->w15 = 0;
  }

  header->w16 = 0;
  header->w17 = 0;
  header->w18 = 0;
  header->w19 = 0;
  header->w20 = 0;
  header->w21 = 0;
  header->w22 = 0;
  header->w23 = 0;

  const bool fault_payload_omitted =
      rtcore_test_memory_fault_payload_omission_enabled();
  if (memory_fault && !fault_payload_omitted) {
    header->w16 = rtcore_compact_result(
        reason, RTCORE_COMPLETION_FLAG_MEMORY_FAULT, completion_seq_low,
        resume_seq_low, window_tag_low);
    header->w17 = RTCORE_FAULT_PAYLOAD_MAGIC;
    header->w18 = RTCORE_FAULT_CODE_TRANSLATION;
    header->w19 = header->lane_slot_index;
    header->w20 = completion_seq_low | (resume_seq_low << 16);
    header->w21 = (unsigned)(header->context_ptr & 0xffffffffu);
    header->w22 = (unsigned)((header->context_ptr >> 32) & 0xffffffffu);
    header->w23 = window_tag_low;
  }

  const bool hit_payload_matches_traversal =
      !closest_hit || !traversal_data.hit_geometry ||
      (header->w9 == (unsigned)traversal_data.closest_hit.geometryType &&
       header->w10 == (unsigned)traversal_data.closest_hit.hitGroupIndex &&
       header->w11 == traversal_data.closest_hit.primitive_index &&
       header->w12 == traversal_data.closest_hit.instance_index &&
       header->w13 ==
           rtcore_float_to_u32(traversal_data.closest_hit.world_min_thit) &&
       header->w14 ==
           rtcore_float_to_u32(
               traversal_data.closest_hit.barycentric_coordinates.x) &&
       header->w15 ==
           rtcore_float_to_u32(
               traversal_data.closest_hit.barycentric_coordinates.y));

  printf("GPGPU-Sim PTX: RT_SUBMIT handoff-dependent-groups (%s:%u), "
         "reason=%s, dispatch={w4=0x%08x,w5=0x%08x,w6=0x%08x,w7=0x%08x}, "
         "hit={w8=0x%08x,w9=0x%08x,w10=0x%08x,w11=0x%08x,"
         "w12=0x%08x,w13=0x%08x,w14=0x%08x,w15=0x%08x}, "
         "resume={w16=0x%08x,w17=0x%08x,w18=0x%08x,w19=0x%08x,"
         "w20=0x%08x,w21=0x%08x,w22=0x%08x,w23=0x%08x}\n",
         pI->source_file(), pI->source_line(),
         rtcore_return_reason_name(reason), header->w4, header->w5,
         header->w6, header->w7, header->w8, header->w9, header->w10,
         header->w11, header->w12, header->w13, header->w14, header->w15,
         header->w16, header->w17, header->w18, header->w19, header->w20,
         header->w21, header->w22, header->w23);
  fflush(stdout);

  if (closest_hit && traversal_data.hit_geometry) {
    printf("GPGPU-Sim PTX: RT_SUBMIT traversal-hit-payload (%s:%u), "
           "source=traversal_data.closest_hit, payload_match=%u, "
           "geometry_type=%u, hit_group_index=%d, primitive_index=%u, "
           "instance_index=%u, world_min_thit_bits=0x%08x, "
           "barycentric_x_bits=0x%08x, barycentric_y_bits=0x%08x, "
           "w9=0x%08x, w10=0x%08x, w11=0x%08x, w12=0x%08x, "
           "w13=0x%08x, w14=0x%08x, w15=0x%08x\n",
           pI->source_file(), pI->source_line(),
           hit_payload_matches_traversal ? 1 : 0,
           (unsigned)traversal_data.closest_hit.geometryType,
           (int)traversal_data.closest_hit.hitGroupIndex,
           traversal_data.closest_hit.primitive_index,
           traversal_data.closest_hit.instance_index, header->w13, header->w14,
           header->w15, header->w9, header->w10, header->w11, header->w12,
           header->w13, header->w14, header->w15);
    fflush(stdout);
    if (!hit_payload_matches_traversal) {
      inst_not_implemented(pI);
      return;
    }
  }

  if (memory_fault) {
    const bool payload_valid =
        (header->w16 & RTCORE_WINDOW_GROUP_VALID) != 0;
    const bool fault_payload_matches =
        payload_valid && ((header->w16 & 0xf) == reason) &&
        (((header->w16 >> 12) & 0xff) == (completion_seq_low & 0xff)) &&
        (((header->w16 >> 20) & 0xff) == (resume_seq_low & 0xff)) &&
        (((header->w16 >> 28) & 0x7) == (window_tag_low & 0x7)) &&
        header->w17 == RTCORE_FAULT_PAYLOAD_MAGIC &&
        header->w18 == RTCORE_FAULT_CODE_TRANSLATION &&
        header->w19 == header->lane_slot_index &&
        header->w20 == (completion_seq_low | (resume_seq_low << 16)) &&
        header->w21 == (unsigned)(header->context_ptr & 0xffffffffu) &&
        header->w22 == (unsigned)((header->context_ptr >> 32) & 0xffffffffu) &&
        header->w23 == window_tag_low;
    printf("GPGPU-Sim PTX: RT_SUBMIT memory-fault-payload (%s:%u), "
           "payload_valid=%u, payload_match=%u, fault_code=%u, "
           "lane_slot_index=%u, context_ptr=0x%llx, "
           "completion_seq=%u, resume_seq=%u, window_tag=%u, "
           "w16=0x%08x, w17=0x%08x, w18=0x%08x, w19=0x%08x, "
           "w20=0x%08x, w21=0x%08x, w22=0x%08x, w23=0x%08x\n",
           pI->source_file(), pI->source_line(), payload_valid ? 1 : 0,
           fault_payload_matches ? 1 : 0, RTCORE_FAULT_CODE_TRANSLATION,
           header->lane_slot_index, header->context_ptr, completion_seq_low,
           resume_seq_low, window_tag_low, header->w16, header->w17,
           header->w18, header->w19, header->w20, header->w21, header->w22,
           header->w23);
    fflush(stdout);
  }
}

bool rtcore_submit_operands_are_valid(unsigned long long context_ptr,
                                      unsigned long long handoff_window_base) {
  return context_ptr != 0 && handoff_window_base != 0 &&
         (context_ptr % RTCORE_CONTEXT_ALIGNMENT) == 0 &&
         (handoff_window_base % RTCORE_HANDOFF_WINDOW_ALIGNMENT) == 0;
}

const char *rtcore_return_reason_name(unsigned reason) {
  switch (reason) {
    case RTCORE_RETURN_FOR_CLOSEST_HIT:
      return "closest_hit";
    case RTCORE_RETURN_FOR_MISS:
      return "miss";
    case RTCORE_RETURN_FOR_MEMORY_FAULT:
      return "memory_fault";
    default:
      return "unknown";
  }
}

bool rtcore_software_lazy_load_synthetic_groups(
    const ptx_instruction *pI, const rtcore_synthetic_handoff_header &window,
    unsigned reason, unsigned completion_seq_low, unsigned resume_seq_low,
    unsigned window_tag_low, unsigned lane_slot_index) {
  const bool known_reason =
      reason == RTCORE_RETURN_FOR_CLOSEST_HIT ||
      reason == RTCORE_RETURN_FOR_MISS ||
      reason == RTCORE_RETURN_FOR_MEMORY_FAULT;
  const bool dispatch_required =
      reason == RTCORE_RETURN_FOR_CLOSEST_HIT ||
      reason == RTCORE_RETURN_FOR_MISS;
  const bool dispatch_valid =
      !dispatch_required || (window.w4 & RTCORE_WINDOW_GROUP_VALID) != 0;
  const bool dispatch_reason_matches =
      !dispatch_required || ((window.w4 & 0xf) == reason);
  const bool dispatch_seq_matches =
      !dispatch_required ||
      (((window.w4 >> 12) & 0xff) == (completion_seq_low & 0xff) &&
       (((window.w4 >> 20) & 0xff) == (resume_seq_low & 0xff)));
  const bool dispatch_tag_matches =
      !dispatch_required || (((window.w4 >> 28) & 0x7) == (window_tag_low & 0x7));
  const bool dispatch_reserved_matches =
      !dispatch_required || window.w7 == 0;
  const bool dispatch_aux_matches =
      reason != RTCORE_RETURN_FOR_MISS || window.w6 == 0;

  const bool hit_required = reason == RTCORE_RETURN_FOR_CLOSEST_HIT;
  const bool hit_valid =
      !hit_required || (window.w8 & RTCORE_WINDOW_GROUP_VALID) != 0;
  const bool hit_reason_matches =
      !hit_required || ((window.w8 & 0xf) == reason);
  const bool hit_seq_matches =
      !hit_required ||
      (((window.w8 >> 12) & 0xff) == (completion_seq_low & 0xff) &&
       (((window.w8 >> 20) & 0xff) == (resume_seq_low & 0xff)));
  const bool hit_tag_matches =
      !hit_required || (((window.w8 >> 28) & 0x7) == (window_tag_low & 0x7));
  const bool hit_group_zero_when_unused =
      hit_required || (window.w8 == 0 && window.w9 == 0 && window.w10 == 0 &&
                       window.w11 == 0 && window.w12 == 0 && window.w13 == 0 &&
                       window.w14 == 0 && window.w15 == 0);

  const bool resume_required = false;
  const bool resume_valid = !resume_required;
  const bool resume_seq_matches = !resume_required;
  const bool resume_tag_matches = !resume_required;

  const bool fault_required = reason == RTCORE_RETURN_FOR_MEMORY_FAULT;
  const bool fault_valid =
      !fault_required || (window.w16 & RTCORE_WINDOW_GROUP_VALID) != 0;
  const bool fault_reason_matches =
      !fault_required || ((window.w16 & 0xf) == reason);
  const bool fault_seq_matches =
      !fault_required ||
      (((window.w16 >> 12) & 0xff) == (completion_seq_low & 0xff) &&
       (((window.w16 >> 20) & 0xff) == (resume_seq_low & 0xff)));
  const bool fault_tag_matches =
      !fault_required || (((window.w16 >> 28) & 0x7) == (window_tag_low & 0x7));
  const bool fault_payload_magic_matches =
      !fault_required || window.w17 == RTCORE_FAULT_PAYLOAD_MAGIC;
  const bool fault_code_matches =
      !fault_required || window.w18 == RTCORE_FAULT_CODE_TRANSLATION;
  const bool fault_lane_matches =
      !fault_required || window.w19 == lane_slot_index;
  const bool fault_context_matches =
      !fault_required ||
      (window.w21 == (unsigned)(window.context_ptr & 0xffffffffu) &&
       window.w22 == (unsigned)((window.context_ptr >> 32) & 0xffffffffu));
  const bool fault_payload_matches =
      !fault_required ||
      (fault_valid && fault_reason_matches && fault_seq_matches &&
       fault_tag_matches && fault_payload_magic_matches &&
       fault_code_matches && fault_lane_matches && fault_context_matches &&
       window.w20 == (completion_seq_low | (resume_seq_low << 16)) &&
       window.w23 == window_tag_low);
  const bool resume_reserved =
      fault_required ||
      (window.w16 == 0 && window.w17 == 0 && window.w18 == 0 &&
       window.w19 == 0 && window.w20 == 0 && window.w21 == 0 &&
       window.w22 == 0 && window.w23 == 0);

  const bool accepted =
      known_reason && dispatch_valid && dispatch_reason_matches &&
      dispatch_seq_matches && dispatch_tag_matches &&
      dispatch_reserved_matches && dispatch_aux_matches && hit_valid &&
      hit_reason_matches && hit_seq_matches && hit_tag_matches &&
      hit_group_zero_when_unused && resume_valid && resume_seq_matches &&
      resume_tag_matches && resume_reserved && fault_valid &&
      fault_reason_matches && fault_seq_matches && fault_tag_matches &&
      fault_payload_matches;

  printf("GPGPU-Sim PTX: RT_SUBMIT software-lazy-load (%s:%u), "
         "lane_slot_index=%u, reason=%s, dispatch_required=%u, "
         "dispatch_valid=%u, "
         "dispatch_reason_match=%u, dispatch_seq_match=%u, "
         "dispatch_tag_match=%u, dispatch_reserved_match=%u, "
         "dispatch_aux_match=%u, hit_required=%u, hit_valid=%u, "
         "hit_reason_match=%u, hit_seq_match=%u, hit_tag_match=%u, "
         "hit_unused_zero=%u, resume_required=%u, resume_valid=%u, "
         "resume_seq_match=%u, resume_tag_match=%u, resume_reserved=%u, "
         "fault_required=%u, fault_valid=%u, fault_reason_match=%u, "
         "fault_seq_match=%u, fault_tag_match=%u, "
         "fault_payload_magic_match=%u, fault_code_match=%u, "
         "fault_lane_match=%u, fault_context_match=%u, "
         "fault_payload_match=%u, "
         "accepted=%u\n",
         pI->source_file(), pI->source_line(), lane_slot_index,
         rtcore_return_reason_name(reason), dispatch_required ? 1 : 0,
         dispatch_valid ? 1 : 0, dispatch_reason_matches ? 1 : 0,
         dispatch_seq_matches ? 1 : 0, dispatch_tag_matches ? 1 : 0,
         dispatch_reserved_matches ? 1 : 0, dispatch_aux_matches ? 1 : 0,
         hit_required ? 1 : 0, hit_valid ? 1 : 0,
         hit_reason_matches ? 1 : 0, hit_seq_matches ? 1 : 0,
         hit_tag_matches ? 1 : 0, hit_group_zero_when_unused ? 1 : 0,
         resume_required ? 1 : 0, resume_valid ? 1 : 0,
         resume_seq_matches ? 1 : 0, resume_tag_matches ? 1 : 0,
         resume_reserved ? 1 : 0, fault_required ? 1 : 0,
         fault_valid ? 1 : 0, fault_reason_matches ? 1 : 0,
         fault_seq_matches ? 1 : 0, fault_tag_matches ? 1 : 0,
         fault_payload_magic_matches ? 1 : 0,
         fault_code_matches ? 1 : 0, fault_lane_matches ? 1 : 0,
         fault_context_matches ? 1 : 0, fault_payload_matches ? 1 : 0,
         accepted ? 1 : 0);
  fflush(stdout);
  return accepted;
}

bool rtcore_software_acquire_synthetic_completion(
    const ptx_instruction *pI, unsigned long long context_ptr,
    unsigned long long handoff_window_base, unsigned lane_slot_index,
    unsigned result_word, ptx_thread_info *thread) {
  const rtcore_synthetic_handoff_key key =
      rtcore_make_synthetic_handoff_key(handoff_window_base, lane_slot_index,
                                        thread);
  const rtcore_synthetic_handoff_header *window =
      rtcore_acquire_synthetic_handoff_window(
          g_rtcore_synthetic_handoff_windows, key);
  const bool tracked_window = window != NULL;
  const bool matching_context =
      tracked_window && window->context_ptr == context_ptr;
  const bool owner_tuple_matches =
      tracked_window &&
      rtcore_synthetic_owner_tuple_matches(*window, key, context_ptr, pI,
                                           thread);
  const bool result_matches_w0 = tracked_window && window->w0 == result_word;
  const bool completion_valid =
      (result_word & RTCORE_COMPLETION_VALID) != 0 &&
      tracked_window && (window->w0 & RTCORE_COMPLETION_VALID) != 0 &&
      (window->w1 & 1u) != 0;

  const unsigned result_reason = result_word & 0xf;
  const unsigned window_reason =
      tracked_window ? ((window->w1 >> 8) & 0xf) : 0;
  const bool reason_matches =
      tracked_window && result_reason == window_reason;

  const unsigned result_flags = (result_word >> 4) & 0xff;
  const unsigned window_flags =
      tracked_window ? ((window->w1 >> 16) & 0xff) : 0;
  const bool flags_matches = tracked_window && result_flags == window_flags;

  const unsigned result_completion_seq_low = (result_word >> 12) & 0xff;
  const unsigned window_completion_seq =
      tracked_window ? (window->w2 & 0xffff) : 0;
  const bool completion_seq_matches =
      tracked_window &&
      result_completion_seq_low == (window_completion_seq & 0xff);

  const unsigned result_resume_seq_low = (result_word >> 20) & 0xff;
  const unsigned window_resume_seq =
      tracked_window ? ((window->w2 >> 16) & 0xffff) : 0;
  const bool resume_seq_matches =
      tracked_window && result_resume_seq_low == (window_resume_seq & 0xff);

  const unsigned result_window_tag_low = (result_word >> 28) & 0x7;
  const unsigned window_tag = tracked_window ? (window->w3 & 0x7) : 0;
  const bool window_tag_matches =
      tracked_window && result_window_tag_low == window_tag;
  const bool dependent_groups_match =
      tracked_window &&
      rtcore_software_lazy_load_synthetic_groups(
          pI, *window, result_reason, result_completion_seq_low,
          result_resume_seq_low, result_window_tag_low, lane_slot_index);

  const bool accepted =
      tracked_window && matching_context && owner_tuple_matches &&
      result_matches_w0 && completion_valid && reason_matches &&
      flags_matches && completion_seq_matches && resume_seq_matches &&
      window_tag_matches && dependent_groups_match;

  printf("GPGPU-Sim PTX: RT_SUBMIT software-acquire (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, result=0x%08x, tracked=%u, matching_context=%u, "
         "owner_tuple_match=%u, w0_match=%u, valid=%u, reason_match=%u, "
         "flags_match=%u, completion_seq_match=%u, resume_seq_match=%u, "
         "tag_match=%u, dependent_groups_match=%u, accepted=%u, reason=%s\n",
         pI->source_file(), pI->source_line(), context_ptr,
         handoff_window_base, lane_slot_index, result_word,
         tracked_window ? 1 : 0,
         matching_context ? 1 : 0, owner_tuple_matches ? 1 : 0,
         result_matches_w0 ? 1 : 0,
         completion_valid ? 1 : 0, reason_matches ? 1 : 0,
         flags_matches ? 1 : 0, completion_seq_matches ? 1 : 0,
         resume_seq_matches ? 1 : 0, window_tag_matches ? 1 : 0,
         dependent_groups_match ? 1 : 0,
         accepted ? 1 : 0, rtcore_return_reason_name(result_reason));
  fflush(stdout);
  return accepted;
}

bool rtcore_current_warp_metadata_is_valid(
    const char *op_name, const ptx_instruction *pI, ptx_thread_info *thread,
    const ptx_thread_info::rtcore_current_warp_metadata *metadata,
    unsigned lane_slot_index) {
  const char *metadata_op_name = op_name != NULL ? op_name : "RT_OP";
  const bool has_metadata = metadata != NULL && metadata->valid;
  const bool owner_hw_sid_matches =
      has_metadata && metadata->owner_hw_sid == thread->get_hw_sid();
  const bool warp_id_matches =
      has_metadata && metadata->warp_id == thread->get_hw_wid();
  const bool static_inst_uid_matches =
      has_metadata && metadata->static_inst_uid == pI->uid();
  const bool lane_active =
      has_metadata &&
      rtcore_lane_is_active_in_mask(metadata->active_mask, lane_slot_index);
  const bool accepted =
      has_metadata && owner_hw_sid_matches && warp_id_matches &&
      static_inst_uid_matches && lane_active;

  printf("GPGPU-Sim PTX: %s current-warp-metadata (%s:%u), "
         "warp_uid=%u, warp_id=%u, owner_hw_sid=%u, "
         "active_mask=0x%08x, static_inst_uid=%u, lane_slot_index=%u, "
         "valid=%u, owner_hw_sid_match=%u, warp_id_match=%u, "
         "static_inst_uid_match=%u, lane_active=%u, accepted=%u\n",
         metadata_op_name, pI->source_file(), pI->source_line(),
         has_metadata ? metadata->warp_uid : 0,
         has_metadata ? metadata->warp_id : 0,
         has_metadata ? metadata->owner_hw_sid : 0,
         has_metadata ? metadata->active_mask : 0,
         has_metadata ? metadata->static_inst_uid : 0, lane_slot_index,
         has_metadata ? 1 : 0, owner_hw_sid_matches ? 1 : 0,
         warp_id_matches ? 1 : 0, static_inst_uid_matches ? 1 : 0,
         lane_active ? 1 : 0, accepted ? 1 : 0);
  fflush(stdout);
  return accepted;
}

enum rtcore_traversal_source_provider {
  RTCORE_TRAVERSAL_SOURCE_PROVIDER_LEGACY_FUNCTIONAL = 0,
  RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_STUB = 1,
  RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_REJECT = 2,
  RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_EMPTY = 3,
  RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_DEFAULT_MISS = 4,
  RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_STATS_MISS = 5,
  RTCORE_TRAVERSAL_SOURCE_PROVIDER_UNSUPPORTED = 6
};

static const char *rtcore_traversal_source_provider_name(
    rtcore_traversal_source_provider provider) {
  switch (provider) {
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_LEGACY_FUNCTIONAL:
      return "legacy_functional";
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_STUB:
      return "rtcore_stub";
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_REJECT:
      return "rtcore_reject";
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_EMPTY:
      return "rtcore_empty";
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_DEFAULT_MISS:
      return "rtcore_default_miss";
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_STATS_MISS:
      return "rtcore_stats_miss";
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_UNSUPPORTED:
    default:
      return "unsupported";
  }
}

static rtcore_traversal_source_provider
rtcore_select_traversal_source_provider() {
  const char *value =
      getenv("VULKAN_SIM_RTCORE_TRAVERSAL_SOURCE_PROVIDER");
  if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0 ||
      strcmp(value, "legacy") == 0 ||
      strcmp(value, "legacy_functional") == 0) {
    return RTCORE_TRAVERSAL_SOURCE_PROVIDER_LEGACY_FUNCTIONAL;
  }
  if (strcmp(value, "rtcore_stub") == 0 || strcmp(value, "rtcore") == 0) {
    return RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_STUB;
  }
  if (strcmp(value, "rtcore_reject") == 0 ||
      strcmp(value, "rtcore_supported_reject") == 0) {
    return RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_REJECT;
  }
  if (strcmp(value, "rtcore_empty") == 0 ||
      strcmp(value, "rtcore_accepted_empty") == 0) {
    return RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_EMPTY;
  }
  if (strcmp(value, "rtcore_default_miss") == 0 ||
      strcmp(value, "rtcore_miss") == 0) {
    return RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_DEFAULT_MISS;
  }
  if (strcmp(value, "rtcore_stats_miss") == 0 ||
      strcmp(value, "rtcore_stats_default_miss") == 0) {
    return RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_STATS_MISS;
  }
  return RTCORE_TRAVERSAL_SOURCE_PROVIDER_UNSUPPORTED;
}

struct rtcore_traversal_source_request {
  rtcore_traversal_source_request()
      : pI(NULL),
        thread(NULL),
        context_ptr(0),
        handoff_window_base(0),
        lane_slot_index(0),
        provider(RTCORE_TRAVERSAL_SOURCE_PROVIDER_LEGACY_FUNCTIONAL) {}

  const ptx_instruction *pI;
  ptx_thread_info *thread;
  unsigned long long context_ptr;
  unsigned long long handoff_window_base;
  unsigned lane_slot_index;
  ptx_thread_info::rtcore_current_warp_metadata warp_metadata;
  rtcore_traversal_source_provider provider;
};

static rtcore_traversal_source_request
rtcore_make_traversal_source_request(
    const ptx_instruction *pI, ptx_thread_info *thread,
    unsigned long long context_ptr, unsigned long long handoff_window_base,
    unsigned lane_slot_index,
    const ptx_thread_info::rtcore_current_warp_metadata *warp_metadata) {
  rtcore_traversal_source_request request;
  request.pI = pI;
  request.thread = thread;
  request.context_ptr = context_ptr;
  request.handoff_window_base = handoff_window_base;
  request.lane_slot_index = lane_slot_index;
  if (warp_metadata != NULL) {
    request.warp_metadata = *warp_metadata;
  }
  request.provider = rtcore_select_traversal_source_provider();
  return request;
}

struct rtcore_traversal_work_descriptor {
  rtcore_traversal_work_descriptor()
      : valid(false),
        provider(RTCORE_TRAVERSAL_SOURCE_PROVIDER_LEGACY_FUNCTIONAL),
        context_ptr(0),
        handoff_window_base(0),
        lane_slot_index(0) {}

  bool valid;
  rtcore_traversal_source_provider provider;
  unsigned long long context_ptr;
  unsigned long long handoff_window_base;
  unsigned lane_slot_index;
  ptx_thread_info::rtcore_current_warp_metadata warp_metadata;
};

static rtcore_traversal_work_descriptor
rtcore_make_traversal_work_descriptor(
    const rtcore_traversal_source_request &request) {
  rtcore_traversal_work_descriptor descriptor;
  descriptor.valid = true;
  descriptor.provider = request.provider;
  descriptor.context_ptr = request.context_ptr;
  descriptor.handoff_window_base = request.handoff_window_base;
  descriptor.lane_slot_index = request.lane_slot_index;
  descriptor.warp_metadata = request.warp_metadata;
  return descriptor;
}

enum rtcore_traversal_provider_reject_reason {
  RTCORE_TRAVERSAL_PROVIDER_REJECT_NONE = 0,
  RTCORE_TRAVERSAL_PROVIDER_REJECT_UNSUPPORTED = 1,
  RTCORE_TRAVERSAL_PROVIDER_REJECT_WORK_DESCRIPTOR_REJECTED = 2,
  RTCORE_TRAVERSAL_PROVIDER_REJECT_ACCEPTED_PAYLOAD_MISSING = 3
};

static const char *rtcore_traversal_provider_reject_reason_name(
    rtcore_traversal_provider_reject_reason reason) {
  switch (reason) {
    case RTCORE_TRAVERSAL_PROVIDER_REJECT_ACCEPTED_PAYLOAD_MISSING:
      return "RTCORE_TRAVERSAL_PROVIDER_ACCEPTED_PAYLOAD_MISSING";
    case RTCORE_TRAVERSAL_PROVIDER_REJECT_WORK_DESCRIPTOR_REJECTED:
      return "RTCORE_TRAVERSAL_PROVIDER_WORK_DESCRIPTOR_REJECTED";
    case RTCORE_TRAVERSAL_PROVIDER_REJECT_UNSUPPORTED:
      return "RTCORE_TRAVERSAL_SOURCE_PROVIDER_UNSUPPORTED";
    case RTCORE_TRAVERSAL_PROVIDER_REJECT_NONE:
      return "RTCORE_TRAVERSAL_PROVIDER_REJECT_NONE";
    default:
      return "RTCORE_TRAVERSAL_PROVIDER_REJECT_UNKNOWN";
  }
}

struct rtcore_traversal_provider_response {
  rtcore_traversal_provider_response()
      : provider(RTCORE_TRAVERSAL_SOURCE_PROVIDER_LEGACY_FUNCTIONAL),
        provider_supported(false),
        provider_accepted(false),
        reject_reason(RTCORE_TRAVERSAL_PROVIDER_REJECT_UNSUPPORTED),
        has_traversal_data(false),
        initialized_default_miss(false),
        hit_geometry(false) {
    memset(&traversal_snapshot, 0, sizeof(traversal_snapshot));
  }

  rtcore_traversal_source_provider provider;
  bool provider_supported;
  bool provider_accepted;
  rtcore_traversal_provider_reject_reason reject_reason;
  Traversal_data traversal_snapshot;
  bool has_traversal_data;
  bool initialized_default_miss;
  bool hit_geometry;
};

struct rtcore_traversal_source_snapshot {
  rtcore_traversal_source_snapshot()
      : provider(RTCORE_TRAVERSAL_SOURCE_PROVIDER_LEGACY_FUNCTIONAL),
        provider_supported(true),
        provider_accepted(true),
        reject_reason(RTCORE_TRAVERSAL_PROVIDER_REJECT_NONE),
        has_traversal_data(false),
        initialized_default_miss(false),
        hit_geometry(false) {
    memset(&traversal_snapshot, 0, sizeof(traversal_snapshot));
  }

  rtcore_traversal_source_provider provider;
  bool provider_supported;
  bool provider_accepted;
  rtcore_traversal_provider_reject_reason reject_reason;
  Traversal_data traversal_snapshot;
  bool has_traversal_data;
  bool initialized_default_miss;
  bool hit_geometry;
};

static rtcore_traversal_provider_response
rtcore_make_legacy_functional_traversal_provider_response(
    const rtcore_traversal_source_request &request) {
  const ptx_instruction *pI = request.pI;
  ptx_thread_info *thread = request.thread;
  rtcore_traversal_provider_response response;
  response.provider = request.provider;
  response.provider_supported = true;
  response.provider_accepted = true;
  response.reject_reason = RTCORE_TRAVERSAL_PROVIDER_REJECT_NONE;
  memory_space *mem = thread->get_global_memory();
  response.has_traversal_data =
      !thread->RT_thread_data->traversal_data.empty();
  if (response.has_traversal_data) {
    Traversal_data *device_traversal_data =
        thread->RT_thread_data->traversal_data.back();
    mem->read((mem_addr_t)device_traversal_data,
              sizeof(response.traversal_snapshot),
              &response.traversal_snapshot);
    response.hit_geometry = response.traversal_snapshot.hit_geometry;
  } else {
    response.traversal_snapshot.hit_geometry = false;
    response.traversal_snapshot.current_shader_counter = -1;
    response.traversal_snapshot.current_shader_type = -1;
    response.traversal_snapshot.missIndex = 0;

    Traversal_data *device_traversal_data =
        (Traversal_data *)VulkanRayTracing::gpgpusim_alloc(
            sizeof(Traversal_data));
    mem->write((mem_addr_t)device_traversal_data, sizeof(Traversal_data),
               &response.traversal_snapshot, thread, pI);
    thread->RT_thread_data->all_hit_data.clear();
    thread->RT_thread_data->traversal_data.push_back(device_traversal_data);
    response.initialized_default_miss = true;
    response.hit_geometry = response.traversal_snapshot.hit_geometry;
  }
  return response;
}

static rtcore_traversal_provider_response
rtcore_make_rtcore_stub_traversal_provider_response(
    const rtcore_traversal_work_descriptor &descriptor) {
  rtcore_traversal_provider_response response;
  response.provider = descriptor.provider;
  response.provider_supported = false;
  response.provider_accepted = false;
  response.reject_reason = RTCORE_TRAVERSAL_PROVIDER_REJECT_UNSUPPORTED;
  printf("GPGPU-Sim PTX: RT_SUBMIT traversal-source-rtcore-stub, "
         "provider=%s, context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, warp_uid=%u, active_mask=0x%08x, "
         "rtcore-stub-provider-unimplemented=1\n",
         rtcore_traversal_source_provider_name(descriptor.provider),
         descriptor.context_ptr, descriptor.handoff_window_base,
         descriptor.lane_slot_index, descriptor.warp_metadata.warp_uid,
         descriptor.warp_metadata.active_mask);
  fflush(stdout);
  return response;
}

static rtcore_traversal_provider_response
rtcore_make_rtcore_reject_traversal_provider_response(
    const rtcore_traversal_work_descriptor &descriptor) {
  rtcore_traversal_provider_response response;
  response.provider = descriptor.provider;
  response.provider_supported = true;
  response.provider_accepted = false;
  response.reject_reason = RTCORE_TRAVERSAL_PROVIDER_REJECT_WORK_DESCRIPTOR_REJECTED;
  printf("GPGPU-Sim PTX: RT_SUBMIT traversal-source-rtcore-reject, "
         "provider=%s, context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, warp_uid=%u, active_mask=0x%08x, "
         "rtcore-reject-provider-supported=1, "
         "rtcore-reject-provider-accepted=0\n",
         rtcore_traversal_source_provider_name(descriptor.provider),
         descriptor.context_ptr, descriptor.handoff_window_base,
         descriptor.lane_slot_index, descriptor.warp_metadata.warp_uid,
         descriptor.warp_metadata.active_mask);
  fflush(stdout);
  return response;
}

static rtcore_traversal_provider_response
rtcore_make_rtcore_empty_traversal_provider_response(
    const rtcore_traversal_work_descriptor &descriptor) {
  rtcore_traversal_provider_response response;
  response.provider = descriptor.provider;
  response.provider_supported = true;
  response.provider_accepted = true;
  response.reject_reason = RTCORE_TRAVERSAL_PROVIDER_REJECT_NONE;
  printf("GPGPU-Sim PTX: RT_SUBMIT traversal-source-rtcore-empty, "
         "provider=%s, context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, warp_uid=%u, active_mask=0x%08x, "
         "rtcore-empty-provider-supported=1, "
         "rtcore-empty-provider-accepted=1, "
         "rtcore-empty-provider-payload=0\n",
         rtcore_traversal_source_provider_name(descriptor.provider),
         descriptor.context_ptr, descriptor.handoff_window_base,
         descriptor.lane_slot_index, descriptor.warp_metadata.warp_uid,
         descriptor.warp_metadata.active_mask);
  fflush(stdout);
  return response;
}

static rtcore_traversal_provider_response
rtcore_make_rtcore_default_miss_traversal_provider_response(
    const rtcore_traversal_work_descriptor &descriptor) {
  rtcore_traversal_provider_response response;
  response.provider = descriptor.provider;
  response.provider_supported = true;
  response.provider_accepted = true;
  response.reject_reason = RTCORE_TRAVERSAL_PROVIDER_REJECT_NONE;
  response.traversal_snapshot.hit_geometry = false;
  response.traversal_snapshot.current_shader_counter = -1;
  response.traversal_snapshot.current_shader_type = -1;
  response.traversal_snapshot.missIndex = 0;
  response.initialized_default_miss = true;
  response.hit_geometry = false;
  printf("GPGPU-Sim PTX: RT_SUBMIT traversal-source-rtcore-default-miss, "
         "provider=%s, context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, warp_uid=%u, active_mask=0x%08x, "
         "rtcore-default-miss-provider-supported=1, "
         "rtcore-default-miss-provider-accepted=1, "
         "rtcore-default-miss-provider-payload=1, "
         "rtcore-default-miss-initialized=1, "
         "rtcore-default-miss-hit-geometry=0\n",
         rtcore_traversal_source_provider_name(descriptor.provider),
         descriptor.context_ptr, descriptor.handoff_window_base,
         descriptor.lane_slot_index, descriptor.warp_metadata.warp_uid,
         descriptor.warp_metadata.active_mask);
  fflush(stdout);
  return response;
}

static const unsigned RTCORE_STATS_MISS_NODE_VISITS = 7;
static const unsigned RTCORE_STATS_MISS_PRIMITIVE_TESTS = 3;

static rtcore_traversal_provider_response
rtcore_make_rtcore_stats_miss_traversal_provider_response(
    const rtcore_traversal_work_descriptor &descriptor) {
  rtcore_traversal_provider_response response;
  response.provider = descriptor.provider;
  response.provider_supported = true;
  response.provider_accepted = true;
  response.reject_reason = RTCORE_TRAVERSAL_PROVIDER_REJECT_NONE;
  response.traversal_snapshot.hit_geometry = false;
  response.traversal_snapshot.current_shader_counter = -1;
  response.traversal_snapshot.current_shader_type = -1;
  response.traversal_snapshot.missIndex = 0;
  response.traversal_snapshot.rtcore_node_visits = RTCORE_STATS_MISS_NODE_VISITS;
  response.traversal_snapshot.rtcore_primitive_tests =
      RTCORE_STATS_MISS_PRIMITIVE_TESTS;
  response.initialized_default_miss = true;
  response.hit_geometry = false;
  printf("GPGPU-Sim PTX: RT_SUBMIT traversal-source-rtcore-stats-miss, "
         "provider=%s, context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, warp_uid=%u, active_mask=0x%08x, "
         "rtcore-stats-miss-provider-supported=1, "
         "rtcore-stats-miss-provider-accepted=1, "
         "rtcore-stats-miss-provider-payload=1, "
         "rtcore-stats-miss-initialized=1, "
         "rtcore-stats-miss-hit-geometry=0, "
         "rtcore-stats-miss-node-visits=%u, "
         "rtcore-stats-miss-primitive-tests=%u\n",
         rtcore_traversal_source_provider_name(descriptor.provider),
         descriptor.context_ptr, descriptor.handoff_window_base,
         descriptor.lane_slot_index, descriptor.warp_metadata.warp_uid,
         descriptor.warp_metadata.active_mask, RTCORE_STATS_MISS_NODE_VISITS,
         RTCORE_STATS_MISS_PRIMITIVE_TESTS);
  fflush(stdout);
  return response;
}

static rtcore_traversal_source_snapshot
rtcore_make_traversal_source_snapshot_from_provider_response(
    const rtcore_traversal_provider_response &response) {
  rtcore_traversal_source_snapshot snapshot;
  snapshot.provider = response.provider;
  snapshot.provider_supported = response.provider_supported;
  snapshot.provider_accepted = response.provider_accepted;
  snapshot.reject_reason = response.reject_reason;
  snapshot.traversal_snapshot = response.traversal_snapshot;
  snapshot.has_traversal_data = response.has_traversal_data;
  snapshot.initialized_default_miss = response.initialized_default_miss;
  snapshot.hit_geometry = response.hit_geometry;
  return snapshot;
}

static rtcore_traversal_provider_response
rtcore_make_unsupported_traversal_provider_response(
    const rtcore_traversal_source_request &request) {
  rtcore_traversal_provider_response response;
  response.provider = request.provider;
  response.provider_supported = false;
  response.provider_accepted = false;
  response.reject_reason = RTCORE_TRAVERSAL_PROVIDER_REJECT_UNSUPPORTED;
  return response;
}

static rtcore_traversal_provider_response
rtcore_make_traversal_provider_response(
    const rtcore_traversal_source_request &request) {
  switch (request.provider) {
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_LEGACY_FUNCTIONAL:
      return rtcore_make_legacy_functional_traversal_provider_response(request);
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_STUB: {
      rtcore_traversal_work_descriptor descriptor =
          rtcore_make_traversal_work_descriptor(request);
      return rtcore_make_rtcore_stub_traversal_provider_response(descriptor);
    }
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_REJECT: {
      rtcore_traversal_work_descriptor descriptor =
          rtcore_make_traversal_work_descriptor(request);
      return rtcore_make_rtcore_reject_traversal_provider_response(descriptor);
    }
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_EMPTY: {
      rtcore_traversal_work_descriptor descriptor =
          rtcore_make_traversal_work_descriptor(request);
      return rtcore_make_rtcore_empty_traversal_provider_response(descriptor);
    }
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_DEFAULT_MISS: {
      rtcore_traversal_work_descriptor descriptor =
          rtcore_make_traversal_work_descriptor(request);
      return rtcore_make_rtcore_default_miss_traversal_provider_response(
          descriptor);
    }
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_RTCORE_STATS_MISS: {
      rtcore_traversal_work_descriptor descriptor =
          rtcore_make_traversal_work_descriptor(request);
      return rtcore_make_rtcore_stats_miss_traversal_provider_response(
          descriptor);
    }
    case RTCORE_TRAVERSAL_SOURCE_PROVIDER_UNSUPPORTED:
    default:
      return rtcore_make_unsupported_traversal_provider_response(request);
  }
}

static rtcore_traversal_source_snapshot
rtcore_make_traversal_source_snapshot(
    const rtcore_traversal_source_request &request) {
  rtcore_traversal_provider_response response =
      rtcore_make_traversal_provider_response(request);
  return rtcore_make_traversal_source_snapshot_from_provider_response(response);
}

static bool rtcore_traversal_source_snapshot_has_accepted_payload(
    const rtcore_traversal_source_snapshot &snapshot) {
  return !snapshot.provider_accepted || snapshot.has_traversal_data ||
         snapshot.initialized_default_miss;
}

struct rtcore_traversal_completion_event {
  rtcore_traversal_completion_event()
      : warp_metadata(),
        context_ptr(0),
        handoff_window_base(0),
        lane_slot_index(0),
        window_generation(0),
        completion_seq_low(0),
        resume_seq_low(0),
        window_tag(0),
        reason(0),
        completion_flags(0),
        result_word(0),
        has_traversal_data(false),
        hit_geometry(false) {
    memset(&traversal_snapshot, 0, sizeof(traversal_snapshot));
    memset(&handoff_key, 0, sizeof(handoff_key));
    memset(&token_key, 0, sizeof(token_key));
    memset(&reservation_key, 0, sizeof(reservation_key));
    memset(&header, 0, sizeof(header));
  }

  Traversal_data traversal_snapshot;
  ptx_thread_info::rtcore_current_warp_metadata warp_metadata;
  rtcore_synthetic_handoff_key handoff_key;
  rtcore_symbolic_rt_token_key token_key;
  rtcore_symbolic_rt_token_reservation_key reservation_key;
  unsigned long long context_ptr;
  unsigned long long handoff_window_base;
  unsigned lane_slot_index;
  unsigned window_generation;
  unsigned completion_seq_low;
  unsigned resume_seq_low;
  unsigned window_tag;
  unsigned reason;
  unsigned completion_flags;
  unsigned result_word;
  rtcore_synthetic_handoff_header header;
  bool has_traversal_data;
  bool hit_geometry;
};

static void rtcore_apply_traversal_source_snapshot(
    rtcore_traversal_completion_event *event,
    const rtcore_traversal_source_snapshot &snapshot) {
  if (event == NULL) {
    return;
  }
  event->has_traversal_data = snapshot.has_traversal_data;
  event->traversal_snapshot = snapshot.traversal_snapshot;
  event->hit_geometry = snapshot.hit_geometry;
}

rtcore_pending_traversal_completion
rtcore_make_pending_traversal_completion_from_event(
    const rtcore_traversal_completion_event &event) {
  rtcore_pending_traversal_completion record;
  record.valid = true;
  record.warp_uid = event.warp_metadata.warp_uid;
  record.warp_id = event.warp_metadata.warp_id;
  record.owner_hw_sid = event.warp_metadata.owner_hw_sid;
  record.active_mask = event.warp_metadata.active_mask;
  record.static_inst_uid = event.warp_metadata.static_inst_uid;
  record.lane_slot_index = event.lane_slot_index;
  record.lane_thread_mask =
      rtcore_lane_thread_mask(event.lane_slot_index);
  record.context_ptr = event.context_ptr;
  record.handoff_window_base = event.handoff_window_base;
  record.result_word = event.result_word;
  record.window_generation = event.window_generation;
  record.completion_seq_low = event.completion_seq_low;
  record.resume_seq_low = event.resume_seq_low;
  record.window_tag = event.window_tag;
  record.reason = event.reason;
  record.traversal_snapshot = event.traversal_snapshot;
  return record;
}

rtcore_adapter_completion_publication
rtcore_make_adapter_completion_publication_from_event(
    const rtcore_traversal_completion_event &event) {
  rtcore_adapter_completion_publication publication;
  publication.warp_uid = event.warp_metadata.warp_uid;
  publication.warp_id = event.warp_metadata.warp_id;
  publication.owner_hw_sid = event.warp_metadata.owner_hw_sid;
  publication.active_mask = event.warp_metadata.active_mask;
  publication.static_inst_uid = event.warp_metadata.static_inst_uid;
  publication.lane_slot_index = event.lane_slot_index;
  publication.lane_thread_mask =
      rtcore_lane_thread_mask(event.lane_slot_index);
  publication.node_visits = event.traversal_snapshot.rtcore_node_visits;
  publication.primitive_tests =
      event.traversal_snapshot.rtcore_primitive_tests;
  return publication;
}

static bool rtcore_enqueue_pending_traversal_completion(
    const rtcore_traversal_completion_event &event) {
  if (rtcore_test_adapter_claim_failure_enabled()) {
    return false;
  }

  rtcore_pending_traversal_completion record =
      rtcore_make_pending_traversal_completion_from_event(event);
  const bool test_metadata_mismatch =
      rtcore_test_delayed_completion_metadata_mismatch_enabled();
  if (test_metadata_mismatch) {
    record.owner_hw_sid = event.warp_metadata.owner_hw_sid + 1;
  }

  const std::pair<unsigned, unsigned> key =
      std::make_pair(record.warp_uid, record.lane_slot_index);
  g_rtcore_pending_traversal_completions[key] = record;

  printf("GPGPU-Sim PTX: RT_SUBMIT delayed-completion-enqueue, "
         "warp_uid=%u, warp_id=%u, owner_hw_sid=%u, active_mask=0x%08x, "
         "static_inst_uid=%u, lane_slot_index=%u, "
         "lane_thread_mask=0x%08x, context_ptr=0x%llx, "
         "handoff_window_base=0x%llx, result=0x%08x, "
         "completion_seq=%u, resume_seq=%u, window_tag=%u, reason=%u\n",
         record.warp_uid, record.warp_id, record.owner_hw_sid,
         record.active_mask, record.static_inst_uid,
         record.lane_slot_index, record.lane_thread_mask,
         record.context_ptr, record.handoff_window_base,
         record.result_word, record.completion_seq_low,
         record.resume_seq_low, record.window_tag, record.reason);
  fflush(stdout);
  return true;
}

bool rtcore_build_traversal_completion_event(
    const ptx_instruction *pI, ptx_thread_info *thread,
    unsigned long long context_ptr, unsigned long long handoff_window_base,
    unsigned lane_slot_index, rtcore_traversal_completion_event *event) {
  if (event == NULL) {
    return false;
  }

  *event = rtcore_traversal_completion_event();
  event->context_ptr = context_ptr;
  event->handoff_window_base = handoff_window_base;
  event->lane_slot_index = lane_slot_index;
  event->handoff_key =
      rtcore_make_synthetic_handoff_key(handoff_window_base, lane_slot_index,
                                        thread);
  event->token_key =
      rtcore_make_symbolic_rt_token_key(context_ptr, handoff_window_base,
                                        lane_slot_index, thread);
  thread->get_rtcore_current_warp_metadata(&event->warp_metadata);
  if (!rtcore_current_warp_metadata_is_valid(
          "RT_SUBMIT", pI, thread, &event->warp_metadata, lane_slot_index)) {
    return false;
  }
  event->reservation_key = rtcore_make_symbolic_rt_token_reservation_key(
      event->warp_metadata, context_ptr, handoff_window_base);
  event->window_generation =
      rtcore_peek_next_synthetic_window_generation(event->handoff_key);
  event->completion_seq_low =
      rtcore_synthetic_completion_seq_for_generation(event->window_generation);
  event->resume_seq_low = 0;
  event->window_tag =
      rtcore_synthetic_window_tag_for_generation(event->window_generation);

  rtcore_traversal_source_request source_request =
      rtcore_make_traversal_source_request(
          pI, thread, event->context_ptr, event->handoff_window_base,
          event->lane_slot_index, &event->warp_metadata);
  rtcore_traversal_source_snapshot source_snapshot =
      rtcore_make_traversal_source_snapshot(source_request);
  const bool provider_unsupported = !source_snapshot.provider_supported;
  const bool provider_rejected =
      source_snapshot.provider_supported && !source_snapshot.provider_accepted;
  const bool provider_payload_missing =
      source_snapshot.provider_supported && source_snapshot.provider_accepted &&
      !rtcore_traversal_source_snapshot_has_accepted_payload(source_snapshot);
  if (provider_payload_missing) {
    source_snapshot.reject_reason = RTCORE_TRAVERSAL_PROVIDER_REJECT_ACCEPTED_PAYLOAD_MISSING;
  }
  if (provider_unsupported || provider_rejected || provider_payload_missing) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=%s, "
           "provider=%s, context_ptr=0x%llx, handoff_window_base=0x%llx, "
           "lane_slot_index=%u, provider_supported=%u, "
           "provider_accepted=%u, traversal-source-provider-unsupported=%u, "
           "traversal-source-provider-rejected=%u, "
           "traversal-source-provider-payload-missing=%u\n",
           pI->source_file(), pI->source_line(),
           rtcore_traversal_provider_reject_reason_name(
               source_snapshot.reject_reason),
           rtcore_traversal_source_provider_name(source_snapshot.provider),
           event->context_ptr, event->handoff_window_base,
           event->lane_slot_index,
           source_snapshot.provider_supported ? 1 : 0,
           source_snapshot.provider_accepted ? 1 : 0,
           provider_unsupported ? 1 : 0, provider_rejected ? 1 : 0,
           provider_payload_missing ? 1 : 0);
    fflush(stdout);
    return false;
  }
  rtcore_apply_traversal_source_snapshot(event, source_snapshot);

  const bool forced_memory_fault =
      rtcore_test_memory_fault_publication_enabled();
  event->reason =
      forced_memory_fault
          ? RTCORE_RETURN_FOR_MEMORY_FAULT
          : (event->hit_geometry ? RTCORE_RETURN_FOR_CLOSEST_HIT
                                 : RTCORE_RETURN_FOR_MISS);
  event->completion_flags =
      forced_memory_fault ? RTCORE_COMPLETION_FLAG_MEMORY_FAULT
                          : RTCORE_COMPLETION_FLAG_TRACE_DONE;
  event->result_word = rtcore_compact_result(
      event->reason, event->completion_flags, event->completion_seq_low,
      event->resume_seq_low, event->window_tag);

  memset(&event->header, 0, sizeof(event->header));
  event->header.context_ptr = event->context_ptr;
  event->header.window_generation = event->window_generation;
  event->header.completion_seq = event->completion_seq_low;
  event->header.resume_seq = event->resume_seq_low;
  event->header.window_tag = event->window_tag;
  event->header.w0 = event->result_word;
  event->header.w1 = rtcore_test_memory_fault_header_omission_enabled()
                         ? 1u
                         : (1u | (event->reason << 8) |
                            (event->completion_flags << 16));
  event->header.w2 = event->completion_seq_low |
                     (event->resume_seq_low << 16);
  event->header.w3 = event->window_tag;
  rtcore_populate_synthetic_owner_tuple(
      &event->header, event->handoff_key, thread,
      event->warp_metadata.active_mask);
  rtcore_publish_synthetic_dependent_groups(
      pI, event->traversal_snapshot, event->reason,
      event->completion_seq_low, event->resume_seq_low, event->window_tag,
      &event->header);
  return rtcore_synthetic_result_lane_binding_matches(
      pI, event->handoff_key, event->header, event->context_ptr,
      event->result_word);
}

bool rtcore_materialize_traversal_completion_lane_transaction(
    const ptx_instruction *pI, ptx_thread_info *thread,
    const operand_info &result,
    const rtcore_traversal_completion_event &event) {
  if (!rtcore_acquire_symbolic_rt_token(
          pI, event.token_key, event.window_generation,
          event.completion_seq_low, event.resume_seq_low, event.window_tag,
          event.result_word)) {
    return false;
  }
  if (rtcore_test_fail_after_token_acquire_enabled()) {
    rtcore_rollback_symbolic_submit_after_token_acquire(
        pI, event.token_key, event.reservation_key, event.handoff_key);
    return false;
  }
  if (!rtcore_note_symbolic_rt_token_reservation_lane_acquired(
          pI, event.reservation_key, event.lane_slot_index)) {
    return false;
  }
  rtcore_publish_synthetic_handoff_window(
      pI, event.handoff_key, event.header);
  if (rtcore_test_fail_after_handoff_publish_enabled()) {
    rtcore_rollback_symbolic_submit_after_handoff_publish(
        pI, event.token_key, event.reservation_key, event.handoff_key);
    return false;
  }

  ptx_reg_t result_data;
  result_data.u32 = event.result_word;
  thread->set_operand_value(result, result_data, U32_TYPE, thread, pI);
  if (rtcore_test_fail_after_result_write_enabled()) {
    rtcore_rollback_symbolic_submit_after_result_write(
        pI, event.token_key, event.reservation_key, event.handoff_key);
    return false;
  }

  return rtcore_software_acquire_synthetic_completion(
      pI, event.context_ptr, event.handoff_window_base,
      event.lane_slot_index, event.result_word, thread);
}

bool rtcore_publish_or_enqueue_traversal_completion_event(
    const ptx_instruction *pI,
    const rtcore_traversal_completion_event &event) {
  bool adapter_claim_accepted = false;
  if (rtcore_delayed_traversal_completion_enabled()) {
    adapter_claim_accepted =
        rtcore_enqueue_pending_traversal_completion(event);
  } else if (rtcore_test_adapter_claim_failure_enabled()) {
    adapter_claim_accepted = false;
  } else {
    rtcore_adapter_completion_publication publication =
        rtcore_make_adapter_completion_publication_from_event(event);
    adapter_claim_accepted =
        rtcore_publish_adapter_completion_record(publication);
  }

  if (!adapter_claim_accepted) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "reason=RT_TOKEN_ADAPTER_CLAIM_FAILED, context_ptr=0x%llx, "
           "handoff_window_base=0x%llx, lane_slot_index=%u, "
           "adapter_claim_accepted=%u, completed_token=0, "
           "remaining_windows=%zu, remaining_tokens=%zu, "
           "remaining_reservations=%zu, remaining_allocator_live=%zu, "
           "consumed=0\n",
           pI->source_file(), pI->source_line(), event.context_ptr,
           event.handoff_window_base, event.lane_slot_index,
           adapter_claim_accepted ? 1 : 0,
           rtcore_synthetic_handoff_window_count(),
           rtcore_symbolic_rt_token_count(),
           rtcore_symbolic_rt_token_reservation_count(),
           rtcore_symbolic_rt_token_allocator_live_count());
    fflush(stdout);
    return false;
  }
  return true;
}

bool rtcore_complete_traversal_completion_event_token(
    const ptx_instruction *pI,
    const rtcore_traversal_completion_event &event) {
  return rtcore_complete_symbolic_rt_token(
      pI, event.token_key, event.result_word, event.completion_seq_low,
      event.resume_seq_low, event.window_tag);
}

// Adapter boundary from functional traversal results to the architectural
// per-lane result/window/token completion transaction.
void rtcore_traversal_completion_adapter_publish(
    const ptx_instruction *pI, ptx_thread_info *thread,
    const operand_info &result, unsigned long long context_ptr,
    unsigned long long handoff_window_base) {
  const unsigned lane_slot_index = rtcore_lane_slot_index(thread);
  rtcore_traversal_completion_event event;
  if (!rtcore_build_traversal_completion_event(
          pI, thread, context_ptr, handoff_window_base, lane_slot_index,
          &event)) {
    inst_not_implemented(pI);
    return;
  }
  if (!rtcore_materialize_traversal_completion_lane_transaction(
          pI, thread, result, event)) {
    inst_not_implemented(pI);
    return;
  }
  if (!rtcore_publish_or_enqueue_traversal_completion_event(pI, event)) {
    inst_not_implemented(pI);
    return;
  }
  if (!rtcore_complete_traversal_completion_event_token(pI, event)) {
    inst_not_implemented(pI);
    return;
  }

  printf("GPGPU-Sim PTX: RT_SUBMIT traversal-complete (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, reason=%s, has_traversal=%u, hit_geometry=%u, "
         "result=0x%08x, w0=0x%08x, w1=0x%08x, w2=0x%08x, w3=0x%08x, "
         "node_visits=%u, primitive_tests=%u\n",
         pI->source_file(), pI->source_line(), event.context_ptr,
         event.handoff_window_base, event.lane_slot_index,
         rtcore_return_reason_name(event.reason),
         event.has_traversal_data ? 1 : 0, event.hit_geometry ? 1 : 0,
         event.result_word, event.header.w0, event.header.w1,
         event.header.w2, event.header.w3,
         event.traversal_snapshot.rtcore_node_visits,
         event.traversal_snapshot.rtcore_primitive_tests);
  fflush(stdout);
}

}  // namespace

void rt_submit_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 3);
  const operand_info &result = pI->operand_lookup(0);
  const operand_info &context_ptr = pI->operand_lookup(1);
  const operand_info &handoff_window_base = pI->operand_lookup(2);

  ptx_reg_t context_ptr_data =
      thread->get_operand_value(context_ptr, context_ptr, B64_TYPE, thread, 1);
  ptx_reg_t handoff_window_base_data = thread->get_operand_value(
      handoff_window_base, handoff_window_base, B64_TYPE, thread, 1);
  const unsigned lane_slot_index = rtcore_lane_slot_index(thread);

  if (!rtcore_submit_operands_are_valid(context_ptr_data.u64,
                                        handoff_window_base_data.u64)) {
    printf("GPGPU-Sim PTX: RT_SUBMIT fail-closed (%s:%u), "
           "context_ptr=0x%llx, handoff_window_base=0x%llx, "
           "lane_slot_index=%u\n",
           pI->source_file(), pI->source_line(),
           (unsigned long long)context_ptr_data.u64,
           (unsigned long long)handoff_window_base_data.u64,
           lane_slot_index);
    fflush(stdout);
    inst_not_implemented(pI);
    return;
  }

  ptx_thread_info::rtcore_current_warp_metadata current_warp_metadata;
  thread->get_rtcore_current_warp_metadata(&current_warp_metadata);
  if (!rtcore_current_warp_metadata_is_valid(
          "RT_SUBMIT", pI, thread, &current_warp_metadata, lane_slot_index)) {
    rtcore_reject_symbolic_submit(pI);
    return;
  }

  if (!rtcore_symbolic_submit_lane_is_active(
          pI, lane_slot_index, context_ptr_data.u64,
          handoff_window_base_data.u64, current_warp_metadata.active_mask)) {
    rtcore_reject_symbolic_submit(pI);
    return;
  }

  if (!rtcore_submit_handoff_window_class_is_valid(
          pI, context_ptr_data.u64, handoff_window_base_data.u64,
          lane_slot_index)) {
    rtcore_reject_symbolic_submit(pI);
    return;
  }

  const rtcore_symbolic_resource_profile resource_profile =
      rtcore_get_symbolic_resource_profile();
  rtcore_log_symbolic_resource_profile_once(resource_profile);

  if (!rtcore_symbolic_submit_has_capacity(
          pI, resource_profile, lane_slot_index, context_ptr_data.u64,
          handoff_window_base_data.u64)) {
    rtcore_reject_symbolic_submit(pI);
    return;
  }

  const rtcore_synthetic_handoff_key key = rtcore_make_synthetic_handoff_key(
      handoff_window_base_data.u64, lane_slot_index, thread);
  if (!rtcore_symbolic_submit_lane_slot_available(
          pI, key, context_ptr_data.u64)) {
    rtcore_reject_symbolic_submit(pI);
    return;
  }

  const rtcore_symbolic_rt_token_key token_key =
      rtcore_make_symbolic_rt_token_key(context_ptr_data.u64,
                                        handoff_window_base_data.u64,
                                        lane_slot_index, thread);
  if (!rtcore_symbolic_submit_token_available(pI, token_key)) {
    rtcore_reject_symbolic_submit(pI);
    return;
  }

  const rtcore_symbolic_rt_token_reservation_key reservation_key =
      rtcore_make_symbolic_rt_token_reservation_key(
          current_warp_metadata, context_ptr_data.u64,
          handoff_window_base_data.u64);
  if (!rtcore_symbolic_submit_token_reservation_available(
          pI, resource_profile, reservation_key, token_key, lane_slot_index)) {
    rtcore_reject_symbolic_submit(pI);
    return;
  }

  rtcore_traversal_completion_adapter_publish(
      pI, thread, result, context_ptr_data.u64, handoff_window_base_data.u64);
}

void rt_retire_context_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 2);
  const operand_info &context_ptr = pI->operand_lookup(0);
  const operand_info &handoff_window_base = pI->operand_lookup(1);

  ptx_reg_t context_ptr_data =
      thread->get_operand_value(context_ptr, context_ptr, B64_TYPE, thread, 1);
  ptx_reg_t handoff_window_base_data = thread->get_operand_value(
      handoff_window_base, handoff_window_base, B64_TYPE, thread, 1);

  const bool operands_are_valid = rtcore_submit_operands_are_valid(
      context_ptr_data.u64, handoff_window_base_data.u64);
  const unsigned lane_slot_index = rtcore_lane_slot_index(thread);
  ptx_thread_info::rtcore_current_warp_metadata current_warp_metadata;
  thread->get_rtcore_current_warp_metadata(&current_warp_metadata);
  const unsigned lane_thread_mask = rtcore_lane_thread_mask(lane_slot_index);
  const bool test_active_mask_override =
      rtcore_test_inactive_lane_retire_enabled();
  if (test_active_mask_override) {
    current_warp_metadata.active_mask &= ~lane_thread_mask;
  }
  const bool retire_metadata_valid = rtcore_current_warp_metadata_is_valid(
      "RT_RETIRE_CONTEXT", pI, thread, &current_warp_metadata, lane_slot_index);
  const rtcore_synthetic_handoff_key key = rtcore_make_synthetic_handoff_key(
      handoff_window_base_data.u64, lane_slot_index, thread);
  const rtcore_symbolic_rt_token_key token_key =
      rtcore_make_symbolic_rt_token_key(context_ptr_data.u64,
                                        handoff_window_base_data.u64,
                                        lane_slot_index, thread);
  const rtcore_synthetic_handoff_header *window =
      operands_are_valid
          ? rtcore_acquire_synthetic_handoff_window(
                g_rtcore_synthetic_handoff_windows, key)
          : NULL;
  const bool tracked_window = window != NULL;
  const bool matching_context =
      tracked_window && window->context_ptr == context_ptr_data.u64;
  const bool owner_tuple_matches =
      tracked_window &&
      rtcore_synthetic_owner_tuple_matches(*window, key, context_ptr_data.u64,
                                           pI, thread);
  const bool token_can_retire =
      operands_are_valid && retire_metadata_valid && tracked_window &&
      matching_context && owner_tuple_matches &&
      rtcore_symbolic_rt_token_can_retire(pI, token_key);

  if (!operands_are_valid || !retire_metadata_valid || !tracked_window ||
      !matching_context || !owner_tuple_matches || !token_can_retire) {
    printf("GPGPU-Sim PTX: RT_RETIRE_CONTEXT fail-closed (%s:%u), "
           "context_ptr=0x%llx, handoff_window_base=0x%llx, "
           "lane_slot_index=%u, valid=%u, metadata_valid=%u, tracked=%u, "
           "matching_context=%u, owner_tuple_match=%u, token_can_retire=%u\n",
           pI->source_file(), pI->source_line(),
           (unsigned long long)context_ptr_data.u64,
           (unsigned long long)handoff_window_base_data.u64,
           lane_slot_index, operands_are_valid ? 1 : 0,
           retire_metadata_valid ? 1 : 0, tracked_window ? 1 : 0,
           matching_context ? 1 : 0, owner_tuple_matches ? 1 : 0,
           token_can_retire ? 1 : 0);
    fflush(stdout);
    inst_not_implemented(pI);
    return;
  }

  // Functional execution reaches here once per active lane; real window release
  // is deferred until a warp-level allocator exists.
  const bool released_window = rtcore_release_synthetic_handoff_window(key);
  const bool released_token = rtcore_release_symbolic_rt_token(token_key);
  printf("GPGPU-Sim PTX: RT_RETIRE_CONTEXT synthetic-retire (%s:%u), "
         "context_ptr=0x%llx, handoff_window_base=0x%llx, "
         "lane_slot_index=%u, owner_tuple_match=%u, released=%u, "
         "released_token=%u, remaining_windows=%zu, remaining_tokens=%zu, "
         "remaining_allocator_live=%zu\n",
         pI->source_file(), pI->source_line(),
         (unsigned long long)context_ptr_data.u64,
         (unsigned long long)handoff_window_base_data.u64, lane_slot_index,
         owner_tuple_matches ? 1 : 0, released_window ? 1 : 0,
         released_token ? 1 : 0, rtcore_synthetic_handoff_window_count(),
         rtcore_symbolic_rt_token_count(),
         rtcore_symbolic_rt_token_allocator_live_count());
  fflush(stdout);
}

void end_trace_ray_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VulkanRayTracing::endTraceRay(pI, thread);
}

// VkAccelerationStructureKHR* _topLevelAS,
// uint rayFlags,
// uint cullMask,
// uint sbtRecordOffset,
// uint sbtRecordStride,
// uint missIndex,
// float3 origin,
// float Tmin,
// float3 direction,
// float Tmax,
// int payload,
// const ptx_instruction *pI,
// ptx_thread_info *thread

void call_pc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
  static unsigned call_uid_next = 1;

  const operand_info &target = pI->dst();
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1);

  const symbol *return_var_src = NULL;
  const symbol *return_var_dst = NULL;

  gpgpu_sim *gpu = thread->get_gpu();
  unsigned callee_pc = 0, callee_rpc = 0;
  /*if (gpu->simd_model() == POST_DOMINATOR)*/ {
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc);
    assert(callee_pc == thread->get_pc());
  }

  thread->callstack_push_plus(callee_pc + pI->inst_size(), callee_rpc,
                              return_var_src, return_var_dst, call_uid_next++);
  thread->set_npc(target_pc);
}

void call_miss_shader_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  //   if(thread->get_ctaid().x == 0 && thread->get_ctaid().y == 0 && thread->get_ctaid().z == 0)
    VSIM_DPRINTF("gpgpusim: Ray [%d] miss shader implementation\n", thread->get_uid());
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }
  
  // const operand_info &dst = pI->dst();
  
  VulkanRayTracing::callMissShader(pI, thread);
  // printf("calling miss shader\n");

  // ptx_reg_t data;
  // data.u32 = pc;
  // thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

void call_closest_hit_shader_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  //   if(thread->get_ctaid().x == 0 && thread->get_ctaid().y == 0 && thread->get_ctaid().z == 0)
    VSIM_DPRINTF("gpgpusim: Ray [%d] closest hit shader implementation\n", thread->get_uid());
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }
  
  // const operand_info &dst = pI->dst();
  
  VulkanRayTracing::callClosestHitShader(pI, thread);
  // printf("calling closest hit shader\n");
}

void call_intersection_shader_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  //   if(thread->get_ctaid().x == 0 && thread->get_ctaid().y == 0 && thread->get_ctaid().z == 0)
  VSIM_DPRINTF("gpgpusim: intersection shader implementation\n");
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }

  const operand_info &src = pI->operand_lookup(0);
  ptx_reg_t src_data = thread->get_operand_value(src, src, U32_TYPE, thread, 1);
  uint32_t shader_counter = src_data.u32;

  VulkanRayTracing::callIntersectionShader(pI, thread, shader_counter);
}

void call_anyhit_shader_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: anyhit shader implementation\n");
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }

  const operand_info &src = pI->operand_lookup(0);
  ptx_reg_t src_data = thread->get_operand_value(src, src, U32_TYPE, thread, 1);
  uint32_t shader_counter = src_data.u32;

  VulkanRayTracing::callAnyHitShader(pI, thread, shader_counter);
}

void image_deref_store_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: image_deref_store implementation\n");
  if(print_debug_insts)
  {
    printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                      thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                      thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
    fflush(stdout);
  }
  int arg = 0;
  const operand_info &op1 = pI->operand_lookup(arg);
  ptx_reg_t op1_data = thread->get_operand_value(op1, op1, B64_TYPE, thread, 1);
  void *image = (void *)(op1_data.s64);

  arg++;
  const operand_info &op2 = pI->operand_lookup(arg);
  ptx_reg_t op2_data = thread->get_operand_value(op2, op2, U32_TYPE, thread, 1);
  uint32_t gl_LaunchIDEXT_X = (op2_data.u32);

  arg++;
  const operand_info &op3 = pI->operand_lookup(arg);
  ptx_reg_t op3_data = thread->get_operand_value(op3, op3, U32_TYPE, thread, 1);
  uint32_t gl_LaunchIDEXT_Y = (op3_data.u32);

  arg++;
  const operand_info &op4 = pI->operand_lookup(arg);
  ptx_reg_t op4_data = thread->get_operand_value(op4, op4, U32_TYPE, thread, 1);
  uint32_t gl_LaunchIDEXT_Z = (op4_data.u32);

  arg++;
  const operand_info &op5 = pI->operand_lookup(arg);
  ptx_reg_t op5_data = thread->get_operand_value(op5, op5, U32_TYPE, thread, 1);
  uint32_t gl_LaunchIDEXT_W = (op5_data.u32);

  arg++; //MRS_TODO: what is this operand?

  arg++;
  const operand_info &op6 = pI->operand_lookup(arg);
  ptx_reg_t op6_data = thread->get_operand_value(op6, op6, F32_TYPE, thread, 1);
  float hitValue_X = (op6_data.f32);

  arg++;
  const operand_info &op7 = pI->operand_lookup(arg);
  ptx_reg_t op7_data = thread->get_operand_value(op7, op7, F32_TYPE, thread, 1);
  float hitValue_Y = (op7_data.f32);

  arg++;
  const operand_info &op8 = pI->operand_lookup(arg);
  ptx_reg_t op8_data = thread->get_operand_value(op8, op8, F32_TYPE, thread, 1);
  float hitValue_Z = (op8_data.f32);

  arg++;
  const operand_info &op9 = pI->operand_lookup(arg);
  ptx_reg_t op9_data = thread->get_operand_value(op9, op9, F32_TYPE, thread, 1);
  float hitValue_W = (op9_data.f32);

  //MRS_TODO: There are more operands

  VulkanRayTracing::image_store(image, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, gl_LaunchIDEXT_Z, gl_LaunchIDEXT_W,
              hitValue_X, hitValue_Y, hitValue_Z, hitValue_W, pI, thread);
}

void image_deref_load_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: image_deref_load implementation\n");
  ptx_reg_t src0_data, src1_data, src5_data, src6_data, data;
  
  const operand_info &src0 = pI->operand_lookup(0);
  src0_data = thread->get_operand_value(src0, src0, U64_TYPE, thread, 1);
  void* desc = (void*)(src0_data.u64);

  const operand_info &src5 = pI->operand_lookup(5);
  src5_data = thread->get_operand_value(src5, src5, U32_TYPE, thread, 1);
  uint32_t x = src5_data.u32;

  const operand_info &src6 = pI->operand_lookup(6);
  src6_data = thread->get_operand_value(src6, src6, U32_TYPE, thread, 1);
  uint32_t y = src6_data.u32;

  //MRS_TODO: There are more operands

  float c0, c1, c2, c3;

  std::vector<ImageMemoryTransactionRecord> transactions;
  VulkanRayTracing::getTexture(desc, x, y, 0, c0, c1, c2, c3, transactions); // MRS_TODO: x and y are uint

  const operand_info &dst1 = pI->operand_lookup(1);
  const operand_info &dst2 = pI->operand_lookup(2);
  const operand_info &dst3 = pI->operand_lookup(3);
  const operand_info &dst4 = pI->operand_lookup(4);

  data.f32 = c0;
  thread->set_operand_value(dst1, data, F32_TYPE, thread, pI);

  data.f32 = c1;
  thread->set_operand_value(dst2, data, F32_TYPE, thread, pI);

  data.f32 = c2;
  thread->set_operand_value(dst3, data, F32_TYPE, thread, pI);

  data.f32 = c3;
  thread->set_operand_value(dst4, data, F32_TYPE, thread, pI);
}

void store_deref_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

void rt_alloc_mem_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // if(thread->get_tid().x == 0 && thread->get_tid().y == 0 && thread->get_tid().z == 0)
  //   if(thread->get_ctaid().x == 0 && thread->get_ctaid().y == 0 && thread->get_ctaid().z == 0)
    VSIM_DPRINTF("gpgpusim: rt_alloc_mem implementation\n");
    if(print_debug_insts)
    {
      printf("########## running line %d of file %s. thread(%d, %d, %d), cta(%d, %d, %d)\n", pI->source_line(), pI->source_file(),
                                        thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
                                        thread->get_ctaid().x, thread->get_ctaid().y, thread->get_ctaid().z);
      fflush(stdout);
    }
  assert(pI->get_num_operands() == 3);
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  src1_data = thread->get_operand_value(src1, dst, U32_TYPE, thread, 0);
  src2_data = thread->get_operand_value(src2, dst, U32_TYPE, thread, 0);


  std::string name = dst.get_symbol()->name();
  uint32_t size = src1_data.u32;
  nir_variable_mode type = (nir_variable_mode)src2_data.u32;
  uint64_t address = NULL;
  
  // printf("########## variable name = %s, size = %d\n", name.c_str(), size);
  variable_decleration_entry* variable_decleration = thread->RT_thread_data->get_variable_decleration_entry(type, name, size);
  if (variable_decleration != NULL) {
    address = variable_decleration->address;
    if(variable_decleration->type != nir_var_ray_hit_attrib) // MRS_TODO: in raytracing_extended closest hit attribs needs 36 bytes instead of 12 which is wrong
      assert(variable_decleration->size == size);
    assert (address != NULL);

    // // For debugging
    // if (variable_decleration->type == nir_var_ray_hit_attrib) {
    //   memory_space *mem = thread->get_global_memory();
    //   float3 barycentrics;
    //   mem->read(variable_decleration->address, sizeof(float3), &barycentrics);
    //   printf("gpgpusim: Ray [%d] closest hit at (%5.3f, %5.3f, %5.3f)\n", thread->get_uid(), barycentrics.x, barycentrics.y, barycentrics.z);
    // }

  } 
  else {
    address = thread->RT_thread_data->add_variable_decleration_entry(type, name, size);
  }

  data.u64 = address;
  thread->set_operand_value(dst, data, B64_TYPE, thread, pI);
}

void run_anyhit_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  printf("gpgpusim: run_anyhit_impl unimplemented!\n");
  abort();
}

void anyhit_exit_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();
  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, dst, U32_TYPE, thread, 0);
  uint32_t shader_counter = src_data.u32;
  VSIM_DPRINTF("gpgpusim: Ray [%d] anyhit shader_counter %d\n", thread->get_uid(), shader_counter);

  warp_intersection_table* table = VulkanRayTracing::anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
  bool exit_intersection = table->exit_shaders(shader_counter, thread->get_tid().x);

  data.pred =
      (exit_intersection ==
       0);  // inverting predicate since ptxplus uses "1" for a set zero flag
  VSIM_DPRINTF("gpgpusim: anyhit_exit_impl -> %s\n", data.pred ? "anyhit shader" : "no anyhit shader");

  if (!data.pred) {
    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();
    uint32_t n_hits;
    mem->read(&(traversal_data->n_all_hits), sizeof(traversal_data->n_all_hits), &n_hits);

    if (n_hits != 0) {
      VSIM_DPRINTF("gpgpusim: all %d anyhit shader calls complete; processing closest hit!\n", n_hits);
      std::vector<Hit_data*> &hit_list = thread->RT_thread_data->all_hit_data;
      assert(hit_list.size() == n_hits);

      bool hit_exists = false; // TODO: Sync this with accepted closest hits in case some geometry do not use anyhit shader
      float tmin = 0;
      Hit_data hit_attributes;
      Hit_data closest_hit;
      for (unsigned i = 0; i < n_hits; i++) {
        mem->read(hit_list[i], sizeof(Hit_data), &hit_attributes);
        VSIM_DPRINTF("gpgpusim: Ray [%d] (%d) (%5.3f, %5.3f, %5.3f) with t = %5.3f\n", thread->get_uid(), i, hit_attributes.barycentric_coordinates.x, hit_attributes.barycentric_coordinates.y, hit_attributes.barycentric_coordinates.z, hit_attributes.world_min_thit);
        if (hit_attributes.world_min_thit > 0 && // Check the hit is valid
            (tmin == 0 || hit_attributes.world_min_thit < tmin)) // Set as closest hit if no hits yet or hit is closest
        {
          // Closest hit so far
          hit_exists = true;
          closest_hit = hit_attributes;
          tmin = hit_attributes.world_min_thit;
        }
      }

      if (hit_exists) {
        // Update closest hit entry to pass to closest hit shader
        VSIM_DPRINTF("gpgpusim: closest hit identified\n");
        VSIM_DPRINTF("gpgpusim: Ray [%d] closest hit at (%5.3f, %5.3f, %5.3f) with t = %5.3f\n", thread->get_uid(), closest_hit.barycentric_coordinates.x, closest_hit.barycentric_coordinates.y, closest_hit.barycentric_coordinates.z, closest_hit.world_min_thit);

        thread->RT_thread_data->set_hitAttribute(closest_hit.barycentric_coordinates, pI, thread);
        mem->write(&(traversal_data->hit_geometry), sizeof(traversal_data->hit_geometry), &hit_exists, thread, pI);
        mem->write(&(traversal_data->closest_hit), sizeof(Hit_data), &closest_hit, thread, pI);
      }

      else {
        // Update hit_geometry
        VSIM_DPRINTF("gpgpusim: no hits accepted\n");
        mem->write(&(traversal_data->hit_geometry), sizeof(traversal_data->hit_geometry), &hit_exists, thread, pI);
      }
    }
  }
  
  thread->set_operand_value(dst, data, PRED_TYPE, thread, pI);
}

void get_anyhit_shader_data_address_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: get_anyhit_shader_data_address_impl\n");
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();
  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, dst, U32_TYPE, thread, 0);
  uint32_t shader_counter = src_data.u32;

  warp_intersection_table* table = VulkanRayTracing::anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
  void* address = table->get_shader_data_address(shader_counter, thread->get_tid().x);

  data.u64 = (uint64_t)address;
  
  thread->set_operand_value(dst, data, B64_TYPE, thread, pI);
}

void run_intersection_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();
  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, dst, U32_TYPE, thread, 0);
  uint32_t shader_counter = src_data.u32;

  warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
  bool intersection_exists = table->shader_exists(thread->get_tid().x, shader_counter, pI, thread);

  data.pred =
      (intersection_exists ==
       0);  // inverting predicate since ptxplus uses "1" for a set zero flag
  
  thread->set_operand_value(dst, data, PRED_TYPE, thread, pI);
}

void intersection_exit_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();
  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, dst, U32_TYPE, thread, 0);
  uint32_t shader_counter = src_data.u32;

  warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
  bool exit_intersection = table->exit_shaders(shader_counter, thread->get_tid().x);

  data.pred =
      (exit_intersection ==
       0);  // inverting predicate since ptxplus uses "1" for a set zero flag
  VSIM_DPRINTF("gpgpusim: intersection_exit_impl -> %s\n", data.pred ? "intersection shader" : "no intersection shader");
  
  thread->set_operand_value(dst, data, PRED_TYPE, thread, pI);
}

void get_intersection_shader_data_address_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: get_intersection_shader_data_address_impl\n");
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();
  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, dst, U32_TYPE, thread, 0);
  uint32_t shader_counter = src_data.u32;

  warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
  void* address = table->get_shader_data_address(shader_counter, thread->get_tid().x);

  data.u64 = (uint64_t)address;
  
  thread->set_operand_value(dst, data, B64_TYPE, thread, pI);
}

void hit_geometry_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  ptx_reg_t data;

  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();
  bool hit_geometry;
  mem->read(&(traversal_data->hit_geometry), sizeof(traversal_data->hit_geometry), &hit_geometry);

  data.pred =
      (hit_geometry ==
      0);  // inverting predicate since ptxplus uses "1" for a set zero flag

  // Predicate is set to 1 when jumping over the section
  VSIM_DPRINTF("gpgpusim: Ray [%d] hit_geometry_impl -> %s\n", thread->get_uid(), data.pred ? "miss" : "closest hit");

  thread->set_operand_value(dst, data, PRED_TYPE, thread, pI);

}

void get_anyhit_index_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(0);
}

void get_intersection_index_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(0);
}

void get_hitgroup_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  ptx_reg_t data;

  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();
  mem->read(&(traversal_data->closest_hit.hitGroupIndex), sizeof(data.u32), &(data.u32));
  
  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

void get_warp_hitgroup_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(0);
  static uint32_t last_counter = 0;
  static uint32_t last_warp_hitgroup = -1;
  
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();
  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, dst, U32_TYPE, thread, 0);
  uint32_t shader_counter = src_data.u32;

  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

  if(shader_counter == last_counter) {
    data.u32 = last_warp_hitgroup;
  }
  else if(shader_counter == last_counter + 1) {
    last_counter = shader_counter;
    mem->read(&(traversal_data->closest_hit.hitGroupIndex), sizeof(traversal_data->closest_hit.hitGroupIndex), &last_warp_hitgroup);
    data.u32 = last_warp_hitgroup;
  }
  else
    assert(0);
  
  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

void get_closest_hit_shaderID_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static uint32_t last_counter = 0;
  static uint32_t last_warp_hitgroup = -1;
  
  const operand_info &dst = pI->dst();
  ptx_reg_t data;

  memory_space *mem = thread->get_global_memory();
  Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

  VkGeometryTypeKHR geometryType;
  mem->read(&(traversal_data->closest_hit.geometryType), sizeof(traversal_data->closest_hit.geometryType), &geometryType);

  if(geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR)
    data.u32 = *((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt));
  else {
    int32_t hitGroupIndex;
    mem->read(&(traversal_data->closest_hit.hitGroupIndex), sizeof(traversal_data->closest_hit.hitGroupIndex), &hitGroupIndex);

    data.u32 = *((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex);
  }
  
  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

void get_intersection_shaderID_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: get_intersection_shaderID_impl\n");
  static uint32_t last_counter = 0;
  static uint32_t last_warp_hitgroup = -1;

  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();
  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, src, U32_TYPE, thread, 1);
  uint32_t shader_counter = src_data.u32;

  warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
  uint32_t hitGroupIndex = table->get_hitGroupIndex(shader_counter, thread->get_tid().x, pI, thread);

  data.u32 = *((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex + 1);
  
  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

void get_anyhit_shaderID_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  VSIM_DPRINTF("gpgpusim: get_anyhit_shaderID_impl ");
  static uint32_t last_counter = 0;
  static uint32_t last_warp_hitgroup = -1;

  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();
  ptx_reg_t data, src_data;

  src_data = thread->get_operand_value(src, src, U32_TYPE, thread, 1);
  uint32_t shader_counter = src_data.u32;

  warp_intersection_table* table = VulkanRayTracing::anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
  uint32_t hitGroupIndex = table->get_hitGroupIndex(shader_counter, thread->get_tid().x, pI, thread);

  // TODO: Adjust this for situations with both intersection and anyhit shaders
  data.u32 = *((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex + 1);
  VSIM_DPRINTF("shader %d\n", data.u32);
  
  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

// wrap_32_4 %ssa_0, %ssa_0_0, %ssa_0_1, %ssa_0_2, %ssa_0_3
void wrap_32_4_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, src3_data, src4_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  const operand_info &src4 = pI->src4();

  src1_data = thread->get_operand_value(src1, dst, U32_TYPE, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, U32_TYPE, thread, 1);
  src3_data = thread->get_operand_value(src3, dst, U32_TYPE, thread, 1);
  src4_data = thread->get_operand_value(src4, dst, U32_TYPE, thread, 1);

  data.u128.lowest = src1_data.u32;
  data.u128.low = src2_data.u32;
  data.u128.high = src3_data.u32;
  data.u128.highest = src4_data.u32;

  const symbol *name = dst.get_symbol();
  thread->set_reg(name, data);
  // thread->set_operand_value(dst, data, BB128_TYPE, thread, pI);
}

// unwrap_32_4 %ssa_0_0, %ssa_0_1, %ssa_0_2, %ssa_0_3, %ssa_0
void unwrap_32_4_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src_data, data1, data2, data3, data4;

  const operand_info &dst1 = pI->operand_lookup(0);
  const operand_info &dst2 = pI->operand_lookup(1);
  const operand_info &dst3 = pI->operand_lookup(2);
  const operand_info &dst4 = pI->operand_lookup(3);
  const operand_info &src = pI->operand_lookup(4);

  const symbol *name = src.get_symbol();
  src_data = thread->get_reg(name);
  // src_data = thread->get_operand_value(src, src, BB128_TYPE, thread, 1);

  data1.u32 = src_data.u128.lowest;
  data2.u32 = src_data.u128.low;
  data3.u32 = src_data.u128.high;
  data4.u32 = src_data.u128.highest;

  thread->set_operand_value(dst1, data1, U32_TYPE, thread, pI);
  thread->set_operand_value(dst2, data2, U32_TYPE, thread, pI);
  thread->set_operand_value(dst3, data3, U32_TYPE, thread, pI);
  thread->set_operand_value(dst4, data4, U32_TYPE, thread, pI);
}

// get_element_32 %ssa_0_1, %ssa_0, 1
void get_element_32_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  const symbol *name = src1.get_symbol();
  src1_data = thread->get_reg(name);
  // src1_data = thread->get_operand_value(src1, dst, BB128_TYPE, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, U32_TYPE, thread, 1);

  if(src2_data.u32 == 0)
    data.u32 = src1_data.u128.lowest;
  else if(src2_data.u32 == 1)
    data.u32 = src1_data.u128.low;
  else if(src2_data.u32 == 2)
    data.u32 = src1_data.u128.high;
  else if(src2_data.u32 == 3)
    data.u32 = src1_data.u128.highest;
  else
    assert(0);

  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);
}

// set_element_32 %ssa_0, %ssa_0_1, 1
void set_element_32_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  data = thread->get_operand_value(dst, dst, BB128_TYPE, thread, 1);
  src1_data = thread->get_operand_value(src1, dst, U32_TYPE, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, U32_TYPE, thread, 1);

  if(src2_data.u32 == 0)
    data.u128.lowest = src1_data.u32;
  else if(src2_data.u32 == 1)
    data.u128.low = src1_data.u32;
  else if(src2_data.u32 == 2)
    data.u128.high = src1_data.u32;
  else if(src2_data.u32 == 3)
    data.u128.highest = src1_data.u32;
  else
    assert(0);
  
  thread->set_operand_value(dst, data, BB128_TYPE, thread, pI);
}

void shader_clock_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  unsigned long long clock = thread->get_gpu()->gpu_tot_sim_cycle;
  
  ptx_reg_t data;

  data.u32 = clock & ((1ULL << 32) - 1);
  const operand_info &dst = pI->dst();
  thread->set_operand_value(dst, data, U32_TYPE, thread, pI);

  data.u32 = clock >> 32;
  const operand_info &dst2 = pI->operand_lookup(1);
  thread->set_operand_value(dst2, data, U32_TYPE, thread, pI);
}

void copysignf_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  assert(pI->get_num_operands() == 2);

  ptx_reg_t src_data, dst_data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  dst_data = thread->get_operand_value(dst, dst, F32_TYPE, thread, 1);
  src_data = thread->get_operand_value(src1, dst, F32_TYPE, thread, 1);

  if((src_data.f32 > 0 && dst_data.f32 < 0) || (src_data.f32 < 0 && dst_data.f32 > 0))
    dst_data.f32 *= 1;
  
  thread->set_operand_value(dst, dst_data, F32_TYPE, thread, pI);
}
