'''
{
    _id: 'm1',
    name: 'M1',
    skill_id: 'M1',
    instructions: [
        '1.  Sweep* left leg',
        '2.  Sweep* right leg.',
        '3.  Sweep* chest.',
        '4.  Sweep* left arm.',
        '5.  Sweep* right arm.',
        '6.  Look Listen Feel Breathing.',
        '7.  Rake** chest.',
        '8.  Roll over casualty.',
        '9.  Rake the back and buttocks.',
        '10. Check pulse.',
        '11. Check skin temperature / quality.',
        '12. Determine position to leave casualty.'
    ]
}
'''
import os
import glob
import json
import yaml

def main(path='perception/procedural_step_recog/config', out_path='mongo_output'):
    fs = glob.glob(os.path.join(path, '*.yaml'))
    os.makedirs(out_path, exist_ok=True)
    all_skills = {}
    for f in fs:
        try:
            print(f)
            d = yaml.safe_load(open(f).read())

            for sk in d['SKILLS']:
                name = sk['NAME']
                steps = sk['STEPS']
                steps = [f'{i}. {s}' for i, s in enumerate(steps, 1)]
                sid = name.split('-')[0].strip()
                print(name, sid)
                with open(os.path.join(out_path, os.path.splitext(os.path.basename(f))[0]) + '.json', 'w') as fh:
                    json.dump({'_id': sid.lower(), 'name': name, 'skill_id': sid, 'instructions': steps}, fh)

                all_skills[sid] = {'desc': name, 'steps': steps}
        except Exception as e:
            print(e)
            #raise

    with open(f'{out_path}/skills.yaml', 'w') as f:
        f.write(yaml.dump({'step_states': ['unobserved', 'implied', 'done'], 'skills': all_skills}))

import fire
fire.Fire(main)
