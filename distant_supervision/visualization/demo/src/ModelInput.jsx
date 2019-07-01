import React from 'react';
import Button from './components/Button'
import ModelIntro from './components/ModelIntro'
import examples from './examples.json'


// TODO: These are some quickly-accessible examples to try out with your model.  They will get
// added to the select box on the demo page, and will auto-populate your input fields when they
// are selected.  The names here need to match what's read in `handleListChange` below.


let fs = require("fs");
// let examples = JSON.parse(fs.readFileSync('examples.json'));

// TODO: This determines what text shows up in the select box for each example.  The input to
// this function will be one of the items from the `examples` list above.
function summarizeExample(example) {
  return example.e1 + "," + example.e2;
}

// TODO: You can give a model name and description that show up in your demo.
const title = "Your Model Name";
const description = (
  <span>
  If you want a description of what this demo is showing, you can put that here.  Or just leave this
  description empty if you don't need it.
  </span>
);

class ModelInput extends React.Component {
  constructor(props) {
    super(props);
    this.handleListChange = this.handleListChange.bind(this);
    this.onClick = this.onClick.bind(this);
  }

  handleListChange(e) {
    if (e.target.value !== "") {
      // TODO: This gets called when the select box gets changed.  You want to set the values of
      // your input boxes with the content in your examples.
      this.e1.value = examples[e.target.value].e1
      this.e2.value = examples[e.target.value].e2
      this.mentions.value = examples[e.target.value].mentions
      this.rel.value = examples[e.target.value].rel
    }
  }

  onClick() {
    const { runModel } = this.props;

    // TODO: You need to map the values in your input boxes to json values that get sent to your
    // predictor.  The keys in this dictionary need to match what your predictor is expecting to receive.
    runModel({e1: this.e1.value, e2: this.e2.value, mentions: this.mentions.value});
  }

  render() {

    const { outputState } = this.props;

    return (
      <div className="model__content">
        <ModelIntro title={title} description={description} />
        <div className="form__instructions"><span>Enter text or</span>
          <select disabled={outputState === "working"} onChange={this.handleListChange}>
              <option value="">Choose an example...</option>
              {examples.map((example, index) => {
                return (
                    <option value={index} key={index}>{summarizeExample(example) + "..."}</option>
                );
              })}
          </select>
        </div>

       {/*
         * TODO: This is where you add your input fields.  You shouldn't have to change any of the
         * code in render() above here.  We're giving a couple of example inputs here, one for a
         * larger piece of text, like a paragraph (the `textarea`) and one for a shorter piece of
         * text, like a question (the `input`).  You'll probably want to change the variable names
         * here to match the input variable names in your model.
         */}

        <div className="form__field">
          <label>e1</label>
          <input ref={(x) => this.e1 = x} type="text" />
        </div>
        <div className="form__field">
          <label>e2</label>
          <input ref={(x) => this.e2 = x} type="text"/>
        </div>
        <div className="form__field">
          <label>mentions</label>
          <textarea ref={(x) => this.mentions = x} type="text"/>
        </div>

        <div className="form__field">
          <label>rel</label>
          <input ref={(x) => this.rel = x} type="text"/>
        </div>

       {/* You also shouldn't have to change anything below here. */}

        <div className="form__field form__field--btn">
          <Button enabled={outputState !== "working"} onClick={this.onClick} />
        </div>
      </div>
    );
  }
}

export default ModelInput;
