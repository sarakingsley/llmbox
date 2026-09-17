# LLM BOX
LLM Box Software Application for Building Customized and **Affordable** AI Solutions.

# LLMBOX Application Diagram
```
LLMBOX_APP/
│
├──startllm.py      # run this in terminal to start the LLM 
├──modes.py 
├──schema.py
├──structure.py    # creates pydantic models
│
├──configurator/   # application configuration manager
│    ├── main.py
│    ├── config.py
│    ├── rules.py
│    ├── options.py
│    └── output.py
│ 
├──src/             # main application software directory
│    ├── main.py
│    ├── config.py
│    ├── rules.py
│    ├── options.py
│    ├──  output.py
│    └──pydantic_models/
│          ├── personrecords.py
│          └── dogbreeds.py
│
├──models/
│    └───llms/       #place model checkpoints in these directories
│        ├── google/
│        │    └── gemma-3-270m-it/ 
│        ├── microsoft/
│        │   └── phi-4-mini-instruct/
│        ├── cohere/
│        │   └── tiny-aya-global/
│        └── meta/
│            └── llama-3-1-instruct/
│
└──data/
     ├──output/
     │     ├── config_logs/    
     │     ├── chat_logs/           # note: you can create training data 
     │     ├── sys_logs/                    with the chatlogs.
     │     └── experiment_logs/
     │ 
     │
     └──YOUR_DATA/ #place your DATASET in a subdirectory within the data dir.
            ├── original_dataset.json 
            └── dataset_splits/            
                   ├── train_dataset.json  ## split your dataset into train
                   └── test.json           ## & test sets. 
            
```




# Attribution Statement | How to cite this LLM Software Application:
If you use, modify or distribute this lab material, you must properly cite or credit the author.

*Recommended citation:*
```Sara Kingsley. September 2026. LLMBox.```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

### Attribution Requirements
GPL-3.0 requires that copyright notices, license notices, and existing attribution statements be retained when redistributing the software or derivative works. If you modify the software, you must clearly indicate that changes were made. The original authors' copyright information must not be removed.

### What GPL-3.0 Requires

If you distribute this software or a modified version of it, you must:

- Provide a copy of the GPL-3.0 license with the distribution.
- Make the complete corresponding source code available to recipients.
- License any modifications or derivative works under GPL-3.0 as well.
- Clearly document any changes you make to the original code.
- Preserve existing copyright notices and license notices.
- Provide recipients with the same rights to use, study, modify, and redistribute the software that you received.

### Additional Notes

- Commercial use is permitted.
- Private/internal use does not require source code disclosure.
- The software is provided **without warranty** or liability.
- You may not impose additional restrictions that limit the rights granted by the GPL-3.0 license.

For the full license text, see the [LICENSE](le or visit the GNU Project website: https://www.gnu.org/licenses/gpl-3.0.en.html. 【1-d432b6】
