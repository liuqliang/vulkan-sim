// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include "memory.h"
#include <algorithm>
#include <limits>
#include <stdlib.h>
#include "../../libcuda/gpgpu_context.h"
#include "../debug.h"
#include "vulkan_ray_tracing.h"

template <unsigned BSIZE>
memory_space_impl<BSIZE>::memory_space_impl(std::string name,
                                            unsigned hash_size) {
  m_name = name;
  MEM_MAP_RESIZE(hash_size);

  m_log2_block_size = -1;
  for (unsigned n = 0, mask = 1; mask != 0; mask <<= 1, n++) {
    if (BSIZE & mask) {
      assert(m_log2_block_size == (unsigned)-1);
      m_log2_block_size = n;
    }
  }
  assert(m_log2_block_size != (unsigned)-1);
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write_only(mem_addr_t offset, mem_addr_t index,
                                          size_t length, const void *data) {
  m_data[index].write(offset, length, (const unsigned char *)data);
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write_simulator_backing(
    mem_addr_t addr, size_t length, const void *data) {
  size_t nbytes_remain = length;
  size_t src_offset = 0;
  mem_addr_t current_addr = addr;
  while (nbytes_remain > 0) {
    const unsigned offset = current_addr & (BSIZE - 1);
    const mem_addr_t page = current_addr >> m_log2_block_size;
    const size_t tx_bytes =
        std::min((size_t)nbytes_remain, (size_t)BSIZE - offset);
    m_data[page].write(offset, tx_bytes,
                       &((const unsigned char *)data)[src_offset]);
    src_offset += tx_bytes;
    current_addr += tx_bytes;
    nbytes_remain -= tx_bytes;
  }
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::ensure_simulator_backing(mem_addr_t addr,
                                                        size_t length) {
  if (length == 0) {
    return true;
  }
  if (static_cast<mem_addr_t>(length - 1) >
      std::numeric_limits<mem_addr_t>::max() - addr) {
    return false;
  }
  size_t nbytes_remain = length;
  mem_addr_t current_addr = addr;
  while (nbytes_remain > 0) {
    const unsigned offset = current_addr & (BSIZE - 1);
    const mem_addr_t page = current_addr >> m_log2_block_size;
    const size_t tx_bytes =
        std::min(nbytes_remain, static_cast<size_t>(BSIZE - offset));
    m_data[page];
    current_addr += tx_bytes;
    nbytes_remain -= tx_bytes;
  }
  return true;
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::simulator_backing_contains(
    mem_addr_t addr, size_t length) const {
  if (length == 0) {
    return true;
  }
  const mem_addr_t first_page = addr >> m_log2_block_size;
  const mem_addr_t last_page =
      (addr + static_cast<mem_addr_t>(length - 1)) >> m_log2_block_size;
  for (mem_addr_t page = first_page; page <= last_page; ++page) {
    if (m_data.find(page) == m_data.end()) {
      return false;
    }
  }
  return true;
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write(mem_addr_t addr, size_t length,
                                     const void *data,
                                     class ptx_thread_info *thd,
                                     const ptx_instruction *pI) {
  const bool use_simulator_backing =
      use_external_launcher || simulator_backing_contains(addr, length);
  if (!use_simulator_backing) {
    void* vulkan_addr = find_vulkan_buffer(addr);

    if (vulkan_addr) {
      memcpy(vulkan_addr, data, length);
    }
    else {
      printf("gpgpusim: WARNING: Memory backing buffer not found for address %p. This data write may be invalid\n", addr);
      memcpy(addr, data, length);
    }
  }
  else {
    write_simulator_backing(addr, length, data);
    if (!m_watchpoints.empty()) {
      std::map<unsigned, mem_addr_t>::iterator i;
      for (i = m_watchpoints.begin(); i != m_watchpoints.end(); i++) {
        mem_addr_t wa = i->second;
        if (((addr <= wa) && ((addr + length) > wa)) ||
            ((addr > wa) && (addr < (wa + 4))))
          thd->get_gpu()->gpgpu_ctx->the_gpgpusim->g_the_gpu->hit_watchpoint(
              i->first, thd, pI);
      }
    }
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_single_block(mem_addr_t blk_idx,
                                                 mem_addr_t addr, size_t length,
                                                 void *data) const {
  if ((addr + length) > (blk_idx + 1) * BSIZE) {
    printf(
        "GPGPU-Sim PTX: ERROR * access to memory \'%s\' is unaligned : "
        "addr=0x%x, length=%zu\n",
        m_name.c_str(), addr, length);
    printf(
        "GPGPU-Sim PTX: (addr+length)=0x%lx > 0x%x=(index+1)*BSIZE, "
        "index=0x%x, BSIZE=0x%x\n",
        (addr + length), (blk_idx + 1) * BSIZE, blk_idx, BSIZE);
    throw 1;
  }
  typename map_t::const_iterator i = m_data.find(blk_idx);
  if (i == m_data.end()) {
    for (size_t n = 0; n < length; n++)
      ((unsigned char *)data)[n] = (unsigned char)0;
    // printf("GPGPU-Sim PTX:  WARNING reading %zu bytes from unititialized
    // memory at address 0x%x in space %s\n", length, addr, m_name.c_str() );
  } else {
    unsigned offset = addr & (BSIZE - 1);
    unsigned nbytes = length;
    i->second.read(offset, nbytes, (unsigned char *)data);
  }
}

template <unsigned BSIZE>
void *memory_space_impl<BSIZE>::find_vulkan_buffer(mem_addr_t addr) const {
  mem_addr_t index = addr & ~(VULKAN_ADDR_BLK - 1);
  unsigned offset = addr & (VULKAN_ADDR_BLK - 1);

  typename std::map<void *, vulkan_buffer_mapping>::const_iterator mapped =
      m_vulkan_address_map.find((void *)index);
  if (mapped != m_vulkan_address_map.end() &&
      offset < mapped->second.valid_bytes) {
    return (unsigned char *)mapped->second.host_addr + offset;
  } else {
    printf("Could not find %p in Vulkan address map\n", (void*)index);
    return NULL;
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read(mem_addr_t addr, size_t length,
                                    void *data) const {
  const bool use_simulator_backing =
      use_external_launcher || simulator_backing_contains(addr, length);
  if (!use_simulator_backing) {
    void* vulkan_addr = find_vulkan_buffer(addr);

    if (vulkan_addr) {
      memcpy(data, vulkan_addr, length);
    }
    else {
      printf("gpgpusim: WARNING: Memory backing buffer not found for address %p. This data read may be invalid\n", addr);
      memcpy(data, addr, length);
    }
  }
  else {
    read_simulator_backing(addr, length, data);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_simulator_backing(
    mem_addr_t addr, size_t length, void *data) const {
  size_t nbytes_remain = length;
  size_t dst_offset = 0;
  mem_addr_t current_addr = addr;
  while (nbytes_remain > 0) {
    const unsigned offset = current_addr & (BSIZE - 1);
    const mem_addr_t page = current_addr >> m_log2_block_size;
    const size_t tx_bytes =
        std::min((size_t)nbytes_remain, (size_t)BSIZE - offset);
    read_single_block(page, current_addr, tx_bytes,
                      &((unsigned char *)data)[dst_offset]);
    dst_offset += tx_bytes;
    current_addr += tx_bytes;
    nbytes_remain -= tx_bytes;
  }
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::read_vulkan_buffer(mem_addr_t addr,
                                                  size_t length,
                                                  void *data) const {
  if (length == 0) {
    return true;
  }
  if (data == NULL) {
    return false;
  }
  const mem_addr_t first_block_base = addr & ~(VULKAN_ADDR_BLK - 1);
  typename std::map<void *, vulkan_buffer_mapping>::const_iterator first =
      m_vulkan_address_map.find((void *)first_block_base);
  if (first == m_vulkan_address_map.end() ||
      addr < first->second.allocation_base) {
    return false;
  }
  const mem_addr_t allocation_offset = addr - first->second.allocation_base;
  if (allocation_offset > first->second.allocation_size ||
      length > first->second.allocation_size - allocation_offset) {
    return false;
  }

  size_t nbytes_remain = length;
  size_t dst_offset = 0;
  mem_addr_t current_addr = addr;
  while (nbytes_remain > 0) {
    const mem_addr_t block_base = current_addr & ~(VULKAN_ADDR_BLK - 1);
    const size_t block_offset = current_addr & (VULKAN_ADDR_BLK - 1);
    typename std::map<void *, vulkan_buffer_mapping>::const_iterator mapped =
        m_vulkan_address_map.find((void *)block_base);
    if (mapped == m_vulkan_address_map.end() ||
        mapped->second.allocation_base != first->second.allocation_base ||
        block_offset >= mapped->second.valid_bytes) {
      return false;
    }
    const size_t copy_bytes =
        std::min(nbytes_remain, mapped->second.valid_bytes - block_offset);
    memcpy((unsigned char *)data + dst_offset,
           (const unsigned char *)mapped->second.host_addr + block_offset,
           copy_bytes);
    current_addr += copy_bytes;
    dst_offset += copy_bytes;
    nbytes_remain -= copy_bytes;
  }
  return true;
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::print(const char *format, FILE *fout) const {
  typename map_t::const_iterator i_page;

  for (i_page = m_data.begin(); i_page != m_data.end(); ++i_page) {
    fprintf(fout, "%s %08x:", m_name.c_str(), i_page->first);
    i_page->second.print(format, fout);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_watch(addr_t addr, unsigned watchpoint) {
  m_watchpoints[watchpoint] = addr;
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::bind_vulkan_buffer(void *bufferAddr,
                                                  size_t bufferSize,
                                                  void *devPtr) {
  const mem_addr_t allocation_base = (mem_addr_t)devPtr;
  unsigned char *host_base = (unsigned char *)bufferAddr;
  for (size_t offset = 0; offset < bufferSize; offset += VULKAN_ADDR_BLK) {
    vulkan_buffer_mapping mapping;
    mapping.host_addr = host_base + offset;
    mapping.valid_bytes =
        std::min(bufferSize - offset, (size_t)VULKAN_ADDR_BLK);
    mapping.allocation_base = allocation_base;
    mapping.allocation_size = bufferSize;
    m_vulkan_address_map[(void *)(allocation_base + offset)] = mapping;
  }
}

template class memory_space_impl<32>;
template class memory_space_impl<64>;
template class memory_space_impl<8192>;
template class memory_space_impl<16 * 1024>;

void g_print_memory_space(memory_space *mem, const char *format = "%08x",
                          FILE *fout = stdout) {
  mem->print(format, fout);
}

#ifdef UNIT_TEST

int main(int argc, char *argv[]) {
  int errors_found = 0;
  memory_space *mem = new memory_space_impl<32>("test", 4);
  // write address to [address]
  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4)
    mem->write(addr, 4, &addr, NULL, NULL);

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4) {
    unsigned tmp = 0;
    mem->read(addr, 4, &tmp);
    if (tmp != addr) {
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, addr);
    }
  }

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned char val = (addr + 128) % 256;
    mem->write(addr, 1, &val, NULL, NULL);
  }

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned tmp = 0;
    mem->read(addr, 1, &tmp);
    unsigned char val = (addr + 128) % 256;
    if (tmp != val) {
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp,
             (unsigned)val);
    }
  }

  if (errors_found) {
    printf("SUMMARY:  ERRORS FOUND\n");
  } else {
    printf("SUMMARY: UNIT TEST PASSED\n");
  }
}

#endif
