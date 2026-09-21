"""SemIf-compatible evidence/options prompts with explicit OWA semantics."""
import hashlib,json
LABELS=['True','False','Unknown']
DIRECT_SYSTEM=('Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. '
               'Respond with only its uppercase letter, with no explanation or reasoning.')
MATCHED_SYSTEM=('Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. '
                'You may reason privately inside the thinking block. Outside that block, respond with only '
                'the uppercase letter of the chosen option, with no explanation.')
DESCRIPTIONS={
 'True':'The statement follows from the supplied facts and rules.',
 'False':'The explicit negation of the statement follows from the supplied facts and rules.',
 'Unknown':'Neither the statement nor its explicit negation follows from the supplied facts and rules.'}
VERSION='proofwriter-owa-options-v1'

def messages(row,reference=False,oracle=False,order=None):
 order=order or LABELS
 state=row['state']
 if oracle and row['oracle']:
  state+='\nAdditional derived facts:\n'+'\n'.join(x['text'] for x in row['oracle'])
 payload={'evidence':state,'criterion':('Use only these facts and rules, applying rules in their stated direction. '
  'An unstated fact is not automatically false. Determine whether this statement is True, False, or Unknown: '+row['question']),
  'options':[{'letter':chr(65+i),'description':DESCRIPTIONS[x]} for i,x in enumerate(order)]}
 return [{'role':'system','content':DIRECT_SYSTEM if reference else MATCHED_SYSTEM},
         {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]

def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def row_seed(row_id,seed):return int(sha(f'{seed}:{row_id}')[:15],16)

def encode(tokenizer,row,condition,max_tokens,order=None):
 order=order or LABELS
 prompt=tokenizer.apply_chat_template(messages(row,condition=='A-ref',condition=='D',order),tokenize=False,add_generation_prompt=True,enable_thinking=False)
 ids=tokenizer.encode(prompt,add_special_tokens=False)
 slots=[]
 for i in range(len(order)):
  letter=chr(65+i);encoded=tokenizer.encode(letter,add_special_tokens=False)
  if len(encoded)!=1 or tokenizer.decode(encoded)!=letter:raise ValueError('Invalid single-token slot')
  if tokenizer.encode(prompt+letter,add_special_tokens=False)!=ids+encoded:raise ValueError('Answer boundary retokenizes')
  slots+=encoded
 if len(ids)>max_tokens:raise ValueError('Prompt overflow; truncation forbidden')
 return prompt,ids,slots
