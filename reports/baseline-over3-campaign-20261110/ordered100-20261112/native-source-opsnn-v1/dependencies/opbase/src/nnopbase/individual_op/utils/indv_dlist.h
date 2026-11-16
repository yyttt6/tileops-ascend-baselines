/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INDV_DLIST_H_
#define INDV_DLIST_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tagDoubleListNode {
    struct tagDoubleListNode* next;
    struct tagDoubleListNode* pre;
} DoubleListNode;

typedef struct {
    DoubleListNode node;
    unsigned int count;
} DList;

static inline void DoubleListInit(DList* head)
{
    head->node.pre = &(head->node);
    head->node.next = head->node.pre;
    head->count = 0;
}

static inline void DoubleListNodeInit(DoubleListNode* node)
{
    node->pre = nullptr;
    node->next = node->pre;
}

static inline void DoubleListInsertAfter(DoubleListNode* newNode, DoubleListNode* target)
{
    newNode->pre = target;
    newNode->next = target->next;
    target->next = newNode;
    newNode->next->pre = newNode;
}
static inline void DoubleListInsertBefore(DoubleListNode* newNode, DoubleListNode* target)
{
    DoubleListInsertAfter(newNode, target->pre);
}

static inline void DoubleListAddFront(DoubleListNode* newNode, DList* list)
{
    DoubleListInsertAfter(newNode, &(list->node));
    list->count++;
}

static inline void DoubleListAppend(DoubleListNode* newNode, DList* list)
{
    DoubleListInsertBefore(newNode, &(list->node));
    list->count++;
}

static inline void DoubleListRemove(DoubleListNode* node, DList* list)
{
    node->pre->next = node->next;
    node->next->pre = node->pre;
    list->count--;
}

static inline int DoubleListIsEmpty(const DList* const list) { return list->node.next == &(list->node); }

static inline unsigned int DoubleListCount(const DList* const list) { return list->count; }

#ifdef __cplusplus
}
#endif

#endif