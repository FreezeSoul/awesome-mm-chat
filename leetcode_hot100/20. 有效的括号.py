# https://leetcode.cn/problems/valid-parentheses/description/?envType=problem-list-v2&envId=2cktkvj

# 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
# 有效字符串需满足：
# 左括号必须用相同类型的右括号闭合。
# 左括号必须以正确的顺序闭合。
# 每个右括号都有一个对应的相同类型的左括号。
# 示例 1：
# 输入：s = "()"
# 输出：true
# 示例 2：
# 输入：s = "()[]{}"
# 输出：true
# 示例 3：
# 输入：s = "(]"
# 输出：false
# 示例 4：
# 输入：s = "([)]"
# 输出：false
# 示例 5：
# 输入：s = "{[]}"
# 输出：true

# 思路：使用栈来解决
# 1. 如果当前字符是左括号，则将其压入栈中
# 2. 如果当前字符是右括号，则检查栈顶元素是否与其匹配
# 3. 如果栈顶元素与当前字符匹配，则将其弹出栈
# 4. 如果栈顶元素与当前字符不匹配，则返回False
# 5. 如果栈为空，则返回True
# 6. 如果栈不为空，则返回False

class Solution:
    def isValid(self, s: str) -> bool:
        valid_list=[]
        for _s in s:
            if _s in ['{',"(","["]:
                valid_list.append(_s)
            else:
                if len(valid_list)==0:
                    return False
                if (_s == '}' and valid_list[-1]=='{') or (_s == ']' and valid_list[-1]=='[') or (_s == ')' and valid_list[-1]=='(') :
                    del valid_list[-1]
                else:
                    valid_list.append(_s)
        if len(valid_list)==0:
            return True
        return False 
